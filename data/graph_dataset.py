import os
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from datasets import Dataset, concatenate_datasets
from data.preprocess_data import UniformEncoder


def load_dataset(args, accelerator):
    all_data_fields = ['node_ids', 'edge_index', 'source'] if args.mode == 'pt' else ['node_ids', 'edge_index', 'question', 'human', 'bot']

    encoder = UniformEncoder(args)
    encoder.initializer()
    
    splits = []
    splits_string = args.data_split
    if splits_string.find(",") != -1:
        splits = [float(s) for s in splits_string.split(",")]
    elif splits_string.find("/") != -1:
        splits = [float(s) for s in splits_string.split("/")]
    else:
        splits = [float(splits_string)]
    while len(splits) < 2:
        splits.append(0.0)
    splits = splits[:2]
    accelerator.print(f'data splits: {splits}')

    files = os.listdir(args.data_dir)
    jsonl_files = [f for f in files if f.endswith('.jsonl')]

    dfs = []
    if args.mode == 'ft':
        dfs_ft = []
    for file in jsonl_files:
        file_name = f"{args.data_dir}/{file}"
        df = pd.read_json(file_name, lines=True)
        if args.mode == 'ft' and 'node_ids' not in df.keys():
            dfs_ft.append(df)
        else:
            dfs.append(df)

    df = pd.concat(dfs)
    dataset = Dataset.from_dict({k: df[k].to_list() for k in df.keys() if k in all_data_fields})
    # shuffle and split
    dataset_split = dataset.train_test_split(train_size=splits[0]/100.0, shuffle=True, seed=args.seed)

    if args.mode == 'pt':
        accelerator.print(dataset_split)
        return dataset_split['train'], dataset_split['test']
    
    else:
        # shuffle each finetune dataset (Java-Python, Python-Java) separately to avoid data leak
        datasets_ft = [Dataset.from_dict({k: df_ft[k].to_list() for k in df_ft.keys() if k in all_data_fields}) for df_ft in dfs_ft]
        dataset_splits_ft = [ds_ft.train_test_split(train_size=99/100.0, shuffle=True, seed=42) for ds_ft in datasets_ft]
        dataset_ft_train = concatenate_datasets([ds_ft['train'] for ds_ft in dataset_splits_ft])
        dataset_ft_valid = concatenate_datasets([ds_ft['test'] for ds_ft in dataset_splits_ft])
        
        accelerator.print('Graph dataset:')
        accelerator.print(dataset_split)
        accelerator.print('Finetune dataset (train):')
        accelerator.print(dataset_ft_train)
        accelerator.print('Finetune dataset (valid):')
        accelerator.print(dataset_ft_valid)

        return dataset_split['train'], dataset_split['test'], dataset_ft_train, dataset_ft_valid