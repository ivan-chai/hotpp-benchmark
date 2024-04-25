import argparse
import numpy as np
import os
import pickle as pkl
import pyarrow as pa
from datasets import load_dataset


NUM_CLASSES = 3
SPLITS = {
    "train": "train",
    "dev": "validation",
    "test": "test"
}


def parse_args():
    parser = argparse.ArgumentParser("Download, prepare and dump dataset to a parquet file.")
    parser.add_argument("root", help="Dataset root")
    return parser.parse_args()


def dump_parquet(data, path):
    # Timestamps are integer values.
    ids = data["seq_idx"]
    timestamps = [np.array(v, dtype=int) for v in data["time_since_start"]]
    labels = [np.array(v, dtype=int) for v in data["type_event"]]
    table = pa.table({"id": ids, "timestamps": timestamps, "labels": labels})
    pa.parquet.write_table(table, path)


def main(args):
    print("Get dataset")
    cache_dir = os.path.join(args.root, "cache")
    if not os.path.isdir(args.root):
        os.mkdir(args.root)
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    dataset = load_dataset("easytpp/retweet", cache_dir=cache_dir)
    for split, name in SPLITS.items():
        print(f"Dump {split}")
        dst_path = os.path.join(args.root, f"{split}.parquet")
        dump_parquet(dataset[name], dst_path)
    print("OK")


if __name__ == "__main__":
    main(parse_args())
