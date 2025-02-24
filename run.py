import time, os, json, math, logging
import numpy as np
import torch
from torch.utils.data import DataLoader
import dgl
# from deepspeed.ops.adam import FusedAdam as AdamW
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import (
    set_seed,
    get_scheduler,
)
from arguments import prepare_args
from data.graph_dataset import load_dataset
from data.preprocess_data import UniformEncoder
from modeling import Model, build_tokenizer
from utils import print_args, accelerate_train, print_with_rank

# get args
os.environ["TOKENIZERS_PARALLELISM"] = "false"
args = prepare_args()

# start accelerator
set_seed(args.seed)
accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

# print and save args in the main process
print_args(args, accelerator)
if accelerator.is_main_process:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:   
        json.dump(args.dict(), f, indent=2)

# prepare logger
logger = get_logger(__name__)
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger.info(accelerator.state, main_process_only=False)

# used in collate function
node_type_embedding = torch.load(args.node_type_embedding)
encoder = UniformEncoder(args)
encoder.initializer()


def collate_fn(instances):
        input_ids, loss_mask, node_ids, edge_index = [], [], [], []
        for instance in instances:
            if 'node_ids' in instance.keys():
                features = encoder.encode_graph(instance)
            else:
                features = encoder.encode_text(instance)
            if features is not None:
                input_ids.append(features['input_ids'])
                loss_mask.append(features['loss_mask'])
                node_ids.append(features.get('node_ids', None))
                edge_index.append(features.get('edge_index', None))

        n = len(input_ids)

        # batch graphs
        if 'node_ids' in instances[0].keys():
            graphs = []
            for i in range(n):
                edges = torch.LongTensor(edge_index[i]).t().contiguous()
                g = dgl.graph((edges[0,:], edges[1,:]), num_nodes=len(node_ids[i]))
                g.ndata['x'] = node_type_embedding[torch.tensor(node_ids[i])]
                graphs.append(g)
            batch = dgl.batch(graphs)
            # edge_index is changed, features remain the same
            # batch.ndata['x']:         (sum(num_nodes), d_embed)
            # batch.edges():            (2, sum(num_edges))
        else:
            batch = None
        
        result_batch = {
            'g': batch,
            'graph_embedding': batch.ndata['x'],
            'batch_num_nodes': batch.batch_num_nodes()
        } if batch else {}

        loss_mask = torch.tensor(loss_mask).long()
        # dynamic padding
        last_one_pos = (loss_mask == 1).long().cumsum(dim=1).argmax(dim=1)
        # get last non-padding position
        max_pos = last_one_pos.max().item() + 1
        
        result_batch['loss_mask'] = loss_mask.float()[:, 1:max_pos].contiguous()
        input_ids = torch.tensor(input_ids).long()
        result_batch['input_ids'] = input_ids[:, :max_pos - 1].contiguous()
        result_batch['labels'] = input_ids[:, 1:max_pos].contiguous()

        return result_batch


def main():
    t0 = time.time()

    # set seed
    set_seed(args.seed)

    # load dataset
    if args.mode == 'pt':
        train_dataset, valid_dataset = load_dataset(args, accelerator)
    else:
        train_dataset, valid_dataset, train_dataset_ft, valid_dataset_ft = load_dataset(args, accelerator)

    t1 = time.time()
    logger.info(f"Dataset loading time: {t1 - t0:.2f}s")
    
    # load model
    tokenizer = build_tokenizer(args)
    model = Model(args, len(tokenizer))

    # print(model.lm.device)
    t2 = time.time()
    logger.info(f"model loading time: {t2 - t1:.2f}s")

    # dataloader
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size, pin_memory=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, collate_fn=collate_fn, 
        batch_size=args.per_device_eval_batch_size, pin_memory=True
    )
    if args.mode == 'ft':
        train_dataloader_ft = DataLoader(
            train_dataset_ft, shuffle=True, collate_fn=collate_fn,
            batch_size=args.per_device_train_batch_size, pin_memory=True
        )
        valid_dataloader_ft = DataLoader(
            valid_dataset_ft, collate_fn=collate_fn, 
            batch_size=args.per_device_eval_batch_size, pin_memory=True
        )
    else:
        train_dataloader_ft, valid_dataloader_ft = None, None
    
    # if finetuning, train all params, else only pretrain GNN and adapter
    if args.mode == 'ft':
        trained_params = model.parameters()
    elif args.mode == 'pt':
        trained_params = [p for p in model.gnn.parameters()] + [p for p in model.adapter.parameters()]
    else:
        raise NotImplementedError()
    optimizer = AdamW(
        trained_params,
        weight_decay=args.weight_decay,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
    )
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) + (len(train_dataloader_ft) if args.mode == 'ft' else 0) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    logger.info(f"{'=='*100}\nbefore accelerator preparation: [dataloader: {len(train_dataloader)}][epochs: {args.num_train_epochs}][total steps: {args.max_train_steps}]\n{'=='*100}")
    if torch.cuda.is_available():
        model, train_dataloader, valid_dataloader, optimizer, lr_scheduler = accelerator.prepare(
            model, train_dataloader, valid_dataloader, optimizer, lr_scheduler
        )
        if args.mode == 'ft':
            train_dataloader_ft, valid_dataloader_ft = accelerator.prepare(train_dataloader_ft, valid_dataloader_ft)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) + (len(train_dataloader_ft) if args.mode == 'ft' else 0) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterward we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    logger.info(f"{'=='*100}\nafter accelerator preparation: [dataloader: {len(train_dataloader)}][epochs: {args.num_train_epochs}][total steps: {args.max_train_steps}]\n{'=='*100}")

    # Train!
    accelerate_train(accelerator,
                     model,
                     train_dataloader,
                     valid_dataloader,
                     train_dataloader_ft,
                     valid_dataloader_ft,
                     optimizer,
                     lr_scheduler,
                     tokenizer,
                     len(train_dataset),
                     args)


if __name__ == "__main__":
    main()