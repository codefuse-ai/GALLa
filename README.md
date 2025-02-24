# GALLa

## Introduction

Repo for the paper [GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding](https://arxiv.org/abs/2409.04183), which aligns LLMs to code structural graphs (e.g. AST, DFG) to enhance their understanding and semantic representation of code.

## Setup

### Environment

GALLa uses a GNN (Graph Neural Network) and an adapter to bridge graphs and LLMs, similar in essence to vision language models such as LLaVA or Qwen-VL. This codebase uses DGL to implement GNN. See `requirements.txt` for more details.

### Models

The training of GALLa consists of three modules: GNN, adapter, and LLM. We use a [DUPLEX](https://arxiv.org/abs/2501.07114) as GNN and a single-layer cross-attention as the adapter by default. We also implement two alternatives: [MagNet](https://arxiv.org/abs/2102.11391) as GNN or an MLP as adapter. To use these alternatives, rename `model_magnet.py` or `model_mlp.py` to `model.py` in the `modeling` folder.

At inference time, only the LLM is used (which means you can use models trained with GALLa just the way you use the base LLM). Currently we support LLaMA-2, LLaMA-3, Phi-1, StarCoder, CodeGen, and Qwen2.5.

### Data

For the first stage (pretraining), only graph data is required (see `data_sample/pretrain.jsonl` for an example). The data should be one or multiple jsonl files stored in a folder, and three fields are required:
- `node_ids`: ids of the nodes in DGL format
- `edge_index`: edge indicies in DGL format
- `source`: the program's source code

In the first stage, the model is trained to recover source code from the graph. You will also need to pass a node embedding matrix (see `configs/config_pretrain.json`), which is a .pth file containing an $N\times d$ tensor, where $N$ is the total number of node types (43 in our case), and $d$ is the node embedding dimension. In our experiments, we used codet5p-embedding to generate these node type embeddings.

For the second stage (instruction finetuning), two types of data are required: graph data and downstream task data, both stored in one or more jsonl files.

Graph data in the second stage should include four fields:
- `node_ids`: same as stage 1
- `edge_index`: same as stage 1
- `question`: a question about the graph
- `bot`: the answer to the question

Downstream task data should include two fields:
- `human`: the question, or prompt
- `bot`: the answer, or response

Examples of these two types of data are provided in `data_sample` folder. You should place these two types of data in the same folder for training.

## Run

First stage:

```
accelerate launch --config_file accelerate_ds_config.yaml run.py --train_config configs/config_pretrain.json
```

Second stage:
```
accelerate launch --config_file accelerate_ds_config.yaml run.py --train_config configs/config_instruction.json
```

Notes:
- `checkpoint` in the second stage's training config should be the path to the checkpoint saved in first stage's training
- LoRA is supported for the second stage. Simply set `lora` to `true` in the config.
- The models saved during second stage training have exactly the same architecture as the base LLM you are using, and can be used in the same way (e.g. with Hugging Face Transformers or VLLM).
- When evaluating the trained models, please format the problems consistently with downstream task training data in stage 2 for best performance.