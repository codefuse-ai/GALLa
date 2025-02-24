from dataclasses import dataclass, asdict
import argparse, json
from typing import List, Union
import torch

@dataclass
class TrainArgs:

    # train data paths on shared FS
    data_dir: Union[str, List[str]]
    node_type_embedding: str

    # output dir for saving adaptors in peft or full ckpts in full-parameter training
    output_dir: str

    # tensorboard dir for saving tensorboard logs
    tb_dir: str

    # pretrained_model_path, on which is the model you want to train
    pretrained_model_path: str

    # whether to load pretrained checkpoint for finetuning
    checkpoint: Union[str, None] = None

    # model type
    model_type: str = 'phi'

    # training mode: "pt" for pretraining, "ft" for instruction finetuning
    mode: str = "ft"

    # graph embedding dimension
    graph_embedding_dim: int = 256
    graph_hidden_dim: int = 1024

    # number of graph node types
    graph_node_types: int = 43
    
    # graph token placeholder
    graph_pad_token: str = "<｜graph_pad｜>"
    graph_token_num: int = 128

    # train/valid/test split
    data_split: str = "99,1,0"

    # mircro train batch size
    per_device_train_batch_size: int = 8

    # micro eval batch size, always same as micro train batch size
    per_device_eval_batch_size: int = 8

    # lora (for stage 2 only)
    lora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16

    # initial lr
    learning_rate: float = 5e-5

    # minimum lr
    min_lr: float = 5e-6

    # weight decay
    weight_decay: float = 0.1

    # gradient_accumulation_steps
    gradient_accumulation_steps: int = 1

    # lr_scheduler_type
    lr_scheduler_type: str = "cosine"

    # num_warmup_steps
    num_warmup_steps: int = 300

    # num_train_epochs
    num_train_epochs: int = 4

    # seed for reproducing
    seed: int = 42

    # seq_length, context length
    seq_length: int = 4096

    # num of steps for logging training loss
    log_interval: int = 10

    # num of steps for saving ckpt
    checkpointing_steps: int = 100

    # num of steps for evaluation(eval_loss), better same as checkpointing steps
    evaluation_steps: int = 100

    # max train steps, if None, depends on num_train_epochs
    max_train_steps: Union[None, int] = None

    # if checkpointing every epoch, maybe True in sst
    epoch_checkpointing: bool = False

    # if early stop when eval loss is not converging in the past early_stopping_stall_num evaluation point 
    early_stopping: bool = True
    early_stopping_stall_num: int = 5
    
    #ATTENTION_CLASSES = { "eager": Normal Attention, "flash_attention_2": FlashAttention2}
    attn_implementation: str = "flash_attention_2"

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
    
def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default=None)
    parsed = parser.parse_args()
    with open(parsed.train_config, 'r') as f:
        train_config = json.load(f)
    
    args = TrainArgs(**train_config)
    if not torch.cuda.is_available():
        args.attn_implementation = 'eager'
    if args.model_type in ['codegen']:
        args.attn_implementation = 'eager'

    return args