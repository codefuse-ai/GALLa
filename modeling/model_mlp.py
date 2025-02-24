import torch
import torch.nn as nn
from .duplex import DUPLEX
from transformers import AutoModelForCausalLM, AutoModel
from utils import count_parameters, print_rank_0, print_highlight, print_rank_0_highlight
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)


class Adapter(nn.Module):
    def __init__(self, args):
        super(Adapter, self).__init__()
        self.args = args

        # the last layer in the GNN is a linear (without activation)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(args.graph_hidden_dim, args.lm_hidden_size),
            nn.SiLU(),
            nn.Linear(args.lm_hidden_size, args.lm_hidden_size),
            nn.SiLU(),
            nn.Linear(args.lm_hidden_size, args.lm_hidden_size),
        )
        print_rank_0(f"Parameters of cross attention: {count_parameters(self.net) / 1e6:.1f}M")
    
    def forward(self, features, batch):
        # features: (sum(num_node), d_embed)
        embeddings = self.net(features)
        return embeddings


class Model(nn.Module):
    def __init__(self, args, vocab):
        super(Model, self).__init__()
        self.num_heads = 8

        # language model
        self.lm = AutoModelForCausalLM.from_pretrained(
            args.pretrained_model_path,
            attn_implementation=args.attn_implementation,
            torch_dtype="auto",
            trust_remote_code=True, 
        )
        self.lm.gradient_checkpointing_enable()
        self.lm.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        if args.model_type in ['starcoder', 'llama3']:
            self.lm.resize_token_embeddings(vocab)
        # lora
        if args.lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
            )
            self.lm = get_peft_model(self.lm, peft_config)
        print_rank_0(f"Parameters of language model: {count_parameters(self.lm) / 1e9:.2f}B")

        # graph model
        self.embed_dim = args.graph_embedding_dim
        class Args:
            dr_rate: float = 0.1
            n_layers: int = 3
            fusion_layer: int = 1
            input_dim: int = args.graph_embedding_dim
            hidden_dim: int = args.graph_hidden_dim
            output_dim: int = args.graph_hidden_dim
            head: int = 1
            fusion =  None

        args_gnn = Args()
        self.gnn = DUPLEX(args_gnn)
        print_rank_0(f"Parameters of GNN: {count_parameters(self.gnn) / 1000:.1f}K")

        # update args
        args.num_heads = self.num_heads
        args.lm_hidden_size = self.lm.config.hidden_size
        self.args = args

        # adapter
        self.adapter = Adapter(args)
        print_rank_0(f"Parameters of adapter (attention + query): {count_parameters(self.adapter) / 1e6:.1f}M")

        if args.checkpoint:
            print_rank_0_highlight(f"Loading exising checkpoint: {args.checkpoint}")
            self.gnn.load_state_dict(torch.load(f"{args.checkpoint}/GNN.pth"))
            self.adapter.load_state_dict(torch.load(f"{args.checkpoint}/adapter.pth"))

    def forward(self, x):
        bs = x['input_ids'].shape[0]

        if 'graph_embedding' in x.keys():
            # embedding -> (bs, l, d_lm), bf16
            if self.args.model_type in ['llama3', 'phi', 'qwen2.5']:
                if not self.args.lora:
                    inputs_embeds = self.lm.model.embed_tokens(x['input_ids'])
                else:
                    inputs_embeds = self.lm.base_model.model.model.embed_tokens(x['input_ids'])
            elif self.args.model_type in ['starcoder', 'codegen']:
                inputs_embeds = self.lm.transformer.wte(x['input_ids'])
            else:
                raise NotImplementedError()

            # x['graph_embedding']:         (sum(num_nodes), d_embed)
            # x.g.edges():                  (2, sum(num_edges))
            embeddings = x['graph_embedding'].to(self.gnn.am_layers[0].attn_l.dtype)

            # GNN -> (sum(num_node), d_embed), bf16
            # features = self.magnet(real=embeddings, imag=embeddings, edge_index=x['edge_index'])
            features = self.gnn(x['g'], embeddings, embeddings)
            
            # adapter -> (sum(num_node), d_embed)
            embeddings = self.adapter(features, x['batch_num_nodes'])

            inputs_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
            idx = torch.nonzero(x['input_ids'].reshape(-1) == self.args.graph_pad_id).squeeze()
            inputs_embeds[idx] = embeddings
            inputs_embeds = inputs_embeds.reshape(bs, -1, inputs_embeds.shape[-1])
            
            # lm
            # print_rank_0('start lm forward')
            outputs = self.lm(inputs_embeds=inputs_embeds,
                            return_dict=True)
            return outputs
        
        else:
            return self.lm(input_ids=x['input_ids'], return_dict=True)
