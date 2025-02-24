import os
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from data.preprocess_data import UniformEncoder


def filter(args):

    encoder = UniformEncoder(args)
    encoder.initializer()
    

    files = os.listdir(args.data_dir)
    jsonl_files = [f for f in files if f.endswith('.jsonl')]

    for file in jsonl_files:
        file_name = f"{args.data_dir}/{file}"
        df = pd.read_json(file_name, lines=True)
        valid_idx = []
        for i in tqdm(range(len(df))):
            if 'node_ids' in df.keys():
                features = encoder.encode_graph(df.loc[i])
            else:
                features = encoder.encode_text(df.loc[i])
            # returns None if too long
            if features is not None:
                valid_idx.append(i)
        valid_idx = np.array(valid_idx)
        len_orig = len(df)
        print(f"{file}: {len_orig} -> ", end='')
        df = df.loc[valid_idx]
        print(f"{len(df)}")
        if len(df) != len_orig:
            print(f'updating {args.data_dir}/{file}')
            df.to_json(f"{args.data_dir}/{file}", orient='records', lines=True)
