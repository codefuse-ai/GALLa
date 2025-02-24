import torch
import torch.nn as nn
# from .magnet import MagNet
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

        # learnable query: (graph_token_num, lm_d)
        self.q = nn.Parameter(torch.randn(args.graph_token_num, args.lm_hidden_size))
        # cross attention
        self.attn = nn.MultiheadAttention(
            embed_dim=args.lm_hidden_size, 
            num_heads=args.num_heads,
            kdim=args.graph_hidden_dim,
            vdim=args.graph_hidden_dim,
            batch_first=True
        )
        print_rank_0(f"Parameters of learnable query: {self.q.numel() / 1e6:.1f}M")
        print_rank_0(f"Parameters of cross attention: {count_parameters(self.attn) / 1e6:.1f}M")
    
    def forward(self, features, batch):
        # reshape features from (sum(num_node), d_embed) to (bs, max(num_node), d_embed)
        features_2d = features.to(self.q.dtype)
        bs = len(batch)
        max_n = batch.max().item()
        features = torch.zeros((bs, max_n, features_2d.shape[-1]), dtype=features_2d.dtype, device=features_2d.device)
        start_idx = 0
        for i in range(bs):
            end_idx = start_idx + batch[i]
            features[i, :batch[i]] = features_2d[start_idx: end_idx]
            start_idx = end_idx

        # adapter -> (bs, num_graph_tokens, d_lm)
        # expand querys to (bs, n_query, d_lm)
        queries = self.q.expand(bs, -1, -1)

        # mask should be shape (bs, S), where S is source seq length
        # note that Pytorch documentation refers to query as "target", and key/value as "source"
        mask = torch.arange(max_n, device=features.device).expand(bs, max_n) < batch.unsqueeze(1)
        mask = ~mask                    # positions set to True are not allowed to attend
            
        embeddings = self.attn(queries, features, features, key_padding_mask=mask, need_weights=False)[0]
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
        if args.model_type in ['starcoder', 'llama3', 'llama2']:
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
            if self.args.model_type in ['llama3', 'phi', 'llama2', 'qwen2.5']:
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
            
            # adapter -> (bs, num_graph_tokens, d_lm)
            embeddings = self.adapter(features, x['batch_num_nodes'])

            # if lora, inputs_embeds will have no grad func and are thus leaft tensors - can't apply in-place operation
            if self.args.lora:
                inputs_embeds = inputs_embeds.clone()

            for i in range(bs):
                # replace graph embedding            
                graph_token_positions = (x['input_ids'][i] == self.args.graph_pad_id).long().nonzero().squeeze()
                pos_start = graph_token_positions.min()
                pos_end = graph_token_positions.max()
                inputs_embeds[i, pos_start:pos_end+1] = embeddings[i]
            
            # lm
            # print_rank_0('start lm forward')
            outputs = self.lm(inputs_embeds=inputs_embeds,
                            return_dict=True)
            return outputs
        
        else:
            return self.lm(input_ids=x['input_ids'], return_dict=True)
