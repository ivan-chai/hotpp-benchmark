import argparse
import math
import os
import pyspark.sql.functions as F
import json
import numpy as np
from datasets import load_dataset
from ptls.preprocessing import PysparkDataPreprocessor
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import DoubleType, IntegerType, ArrayType
from random import Random


TRANSACTIONS_FILES = [
    "train",
    "test"
]

TARGET_FILE = "train_labels"

SEED = 42
VAL_SIZE = 0.0
TEST_SIZE = 0.1


def parse_args():
    parser = argparse.ArgumentParser("Download, prepare and dump dataset to a parquet file.")
    parser.add_argument("--root", help="Dataset root", default="data")
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
        part = load_dataset("dllllb/datascience-bowl2019", name, cache_dir=cache_dir)
        assert len(part.keys()) == 1
        key = next(iter(part.keys()))
        part = dataset2spark(part[key], name, cache_dir).select([
            "installation_id",
            "game_session",
            "timestamp",
            "event_code",
            "event_type",
            "title",
            "world",
            "event_data"
        ])
        dataset = dataset.union(part) if dataset is not None else part

    # Process key == 'correct' in json data.
    udf_function = F.udf(lambda x: int(json.loads(x).get("correct", 2)), IntegerType())
    dataset = dataset.withColumn("correct", udf_function("event_data"))

    dataset = dataset.selectExpr("installation_id as user_id",
                                 "game_session as id",
                                 "timestamp as timestamps",
                                 "event_code as labels",
                                 "event_type as types",
                                 "title",
                                 "world",
                                 "correct")

    # Timestamps to days.
    fmt = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
    dataset = dataset.withColumn("timestamps", F.unix_timestamp("timestamps", fmt) / (24 * 60 * 60))
    return dataset


def get_targets(cache_dir, transactions):
    print(f"Load targets")
    dataset = load_dataset("dllllb/datascience-bowl2019", TARGET_FILE, cache_dir=cache_dir)
    assert len(dataset.keys()) == 1
    key = next(iter(dataset.keys()))
    dataset = dataset2spark(dataset[key], "targets", cache_dir)
    dataset = dataset.selectExpr("installation_id as user_id",
                                 "game_session as id",
                                 "accuracy_group as target")
    assessments = transactions.where((F.col("types") == "Assessment") & (F.col("labels") == 2000))
    assessments = assessments.select("id", "timestamps")
    dataset = dataset.join(assessments, on="id").withColumnRenamed("timestamps", "target_timestamp")
    return dataset  # user_id, id, target_timestamp, target.


def train_val_test_split(transactions, targets):
    """Select test set from the labeled subset of the dataset."""
    data_ids = {row["user_id"] for row in transactions.select("user_id").distinct().collect()}
    labeled_ids = {row["user_id"] for row in targets.select("user_id").distinct().collect()}
    labeled_ids = data_ids & labeled_ids
    unlabeled_ids = data_ids - labeled_ids

    n_clients_val = int(len(data_ids) * VAL_SIZE)
    n_clients_test = int(len(data_ids) * TEST_SIZE)

    labeled_ids = list(sorted(list(labeled_ids)))
    Random(SEED).shuffle(labeled_ids)
    test_ids = set(labeled_ids[:n_clients_test])
    train_ids = list(sorted(set(labeled_ids[n_clients_test:]) | unlabeled_ids))
    Random(SEED + 1).shuffle(train_ids)
    val_ids = set(train_ids[:n_clients_val])
    train_ids = set(train_ids[n_clients_val:])

    testset = transactions.filter(transactions["user_id"].isin(test_ids))
    trainset = transactions.filter(transactions["user_id"].isin(train_ids))
    valset = transactions.filter(transactions["user_id"].isin(val_ids))
    return trainset.persist(), valset.persist(), testset.persist()


def dump_parquet(df, path, n_partitions):
    df.sort(F.col("user_id"), F.col("id")).repartition(n_partitions, "user_id").write.mode("overwrite").parquet(path)


def main(args):
    cache_dir = os.path.join(args.root, "cache")
    if not os.path.isdir(args.root):
        os.mkdir(args.root)
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    transactions = get_transactions(cache_dir)
    targets = get_targets(cache_dir, transactions)  # user_id, id, target_timestamp, target.
    transactions = transactions.drop("id") # user_id, timestamps, labels, types, title, world, correct.

    print("Transform")
    preprocessor = PysparkDataPreprocessor(
        col_id="user_id",
        col_event_time="timestamps",
        event_time_transformation="none",
        cols_category=["labels", "types", "title", "world"],
        cols_identity=["correct"],
        category_transformation="frequency"
    )
    transactions = preprocessor.fit_transform(transactions).persist()

    # Join with targets
    transactions = targets.join(transactions, on="user_id")  # user_id, id, timestamps, labels, types, title, world, correct, target_timestamp, target.

    # Truncate to the target timestamp.
    def get_index(timestamps, target_timestamp):
        return int(np.searchsorted(np.array(timestamps), target_timestamp)) + 1

    udf_function = F.udf(get_index, IntegerType())
    transactions = transactions.withColumn("index", udf_function("timestamps", "target_timestamp")).drop("target_timestamp")
    cols_to_slice = ["labels", "types", "title", "world", "correct"]
    udf_function = F.udf(lambda seq, index: seq[0:index], ArrayType(IntegerType()))
    for col in cols_to_slice:
        transactions = transactions.withColumn(col, udf_function(col, "index"))
    udf_function = F.udf(lambda seq, index: seq[0: index], ArrayType(DoubleType()))
    transactions = transactions.withColumn("timestamps", udf_function("timestamps", "index"))
    transactions = transactions.drop("index")

    # ID to integer.
    id_mapping = transactions.select("id").distinct()
    window = Window().orderBy("id")
    id_mapping = id_mapping.withColumn("id_int", F.row_number().over(window)).persist()
    dump_parquet(id_mapping, "data/mappping.parquet", 1)
    transactions = transactions.join(id_mapping, on="id").drop("id").withColumnRenamed("id_int", "id")
    targets = targets.join(id_mapping, on="id").drop("id").withColumnRenamed("id_int", "id")

    print("Split & dump")
    train, val, test = train_val_test_split(transactions, targets)
    train_path = os.path.join(args.root, "train.parquet")
    val_path = os.path.join(args.root, "val.parquet")
    test_path = os.path.join(args.root, "test.parquet")
    print(f"Dump train with {train.count()} records to {train_path}")
    dump_parquet(train, train_path, n_partitions=32)
    print(f"Dump val with {val.count()} records to {val_path}")
    dump_parquet(val, val_path, n_partitions=1)
    print(f"Dump test with {test.count()} records to {test_path}")
    dump_parquet(test, test_path, n_partitions=1)
    print("OK")


if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    main(parse_args())
