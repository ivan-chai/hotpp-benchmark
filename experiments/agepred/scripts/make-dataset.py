import argparse
import os
import pyspark.pandas
import pandas as pd
from datasets import load_dataset
from ptls.preprocessing import PysparkDataPreprocessor
from pyspark.sql import SparkSession
from random import Random


TRANSACTIONS_FILES = [
    "transactions_train",
    "transactions_test"
]

TARGET_FILE = "train_target"

SEED = 42
DEV_SIZE = 0.05
TEST_SIZE = 0.1


def parse_args():
    parser = argparse.ArgumentParser("Download, prepare and dump dataset to a parquet file.")
    parser.add_argument("root", help="Dataset root")
    return parser.parse_args()


def dataset2spark(dataset, name, cache_dir):
    spark = SparkSession.builder.getOrCreate()
    path = os.path.join(cache_dir, f"convert-{name}.parquet")
    dataset.to_parquet(path)
    dataset = spark.read.parquet(path)
    return dataset


def get_transactions(cache_dir):
    dataset = None
    for name in TRANSACTIONS_FILES:
        print(f"Load {name}")
        part = load_dataset("dllllb/age-group-prediction", name, cache_dir=cache_dir)
        assert len(part.keys()) == 1
        key = next(iter(part.keys()))
        part = dataset2spark(part[key], name, cache_dir)
        dataset = dataset.union(part) if dataset is not None else part
    dataset = dataset.selectExpr("client_id as id",
                                 "trans_date as timestamps",
                                 "small_group as labels")
    return dataset


def get_targets(cache_dir):
    print(f"Load targets")
    dataset = load_dataset("dllllb/age-group-prediction", TARGET_FILE, cache_dir=cache_dir)
    assert len(dataset.keys()) == 1
    key = next(iter(dataset.keys()))
    dataset = dataset2spark(dataset[key], "targets", cache_dir)
    dataset = dataset.selectExpr("client_id as id",
                                 "bins as target")
    return dataset


def train_dev_test_split(transactions, targets):
    """Select test set from the labeled subset of the dataset."""
    data_ids = {row["id"] for row in transactions.select("id").distinct().collect()}
    labeled_ids = {row["id"] for row in targets.select("id").distinct().collect()}
    labeled_ids = data_ids & labeled_ids
    unlabeled_ids = data_ids - labeled_ids

    labeled_ids = list(sorted(list(labeled_ids)))
    Random(SEED).shuffle(labeled_ids)
    n_clients_test = int(len(labeled_ids) * TEST_SIZE)
    test_ids = set(labeled_ids[-n_clients_test:])
    train_ids = list(sorted(set(labeled_ids[:-n_clients_test]) | unlabeled_ids))
    Random(SEED + 1).shuffle(train_ids)
    n_clients_dev = int(len(train_ids) * DEV_SIZE)
    dev_ids = set(train_ids[-n_clients_dev:])
    train_ids = set(train_ids[:-n_clients_dev])

    testset = transactions.filter(transactions["id"].isin(test_ids))
    testset = testset.join(targets, on="id", how="inner")
    trainset = transactions.filter(transactions["id"].isin(train_ids))
    devset = transactions.filter(transactions["id"].isin(dev_ids))
    return trainset.persist(), devset.persist(), testset.persist()


def dump_parquet(df, path):
    df.write.mode("overwrite").parquet(path)


def main(args):
    cache_dir = os.path.join(args.root, "cache")
    if not os.path.isdir(args.root):
        os.mkdir(args.root)
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    transactions = get_transactions(cache_dir)
    targets = get_targets(cache_dir)

    print("Transform")
    preprocessor = PysparkDataPreprocessor(
        col_id="id",
        col_event_time="timestamps",
        event_time_transformation="none",
        cols_category=["labels"],
        category_transformation="frequency"
    )
    transactions = preprocessor.fit_transform(transactions).persist()

    print("Split")
    train, dev, test = train_dev_test_split(transactions, targets)

    train_path = os.path.join(args.root, "train.parquet")
    dev_path = os.path.join(args.root, "dev.parquet")
    test_path = os.path.join(args.root, "test.parquet")
    print(f"Dump train with {train.count()} records to {train_path}")
    dump_parquet(train, train_path)
    print(f"Dump dev with {dev.count()} records to {dev_path}")
    dump_parquet(dev, dev_path)
    print(f"Dump test with {test.count()} records to {test_path}")
    dump_parquet(test, test_path)
    print("OK")


if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    main(parse_args())
