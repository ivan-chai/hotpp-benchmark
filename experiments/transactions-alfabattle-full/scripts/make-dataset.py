import argparse
import math
import os
import pyspark.sql.functions as F
from datasets import load_dataset
from ptls.preprocessing import PysparkDataPreprocessor
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import ArrayType, FloatType, IntegerType
from random import Random


TRANSACTIONS_FILES = [
    "test_transactions",
    "train_transactions"
]

TARGET_FILE = "train_target"

SEED = 42
VAL_SIZE = 0.05
TEST_SIZE = 0.1


def parse_args():
    parser = argparse.ArgumentParser("Download, prepare and dump dataset to a parquet file.")
    parser.add_argument("--root", help="Dataset root", default="data")
    return parser.parse_args()


def dataset2spark(dataset, name, cache_dir):
    spark = SparkSession.builder.getOrCreate()
    path = os.path.join(cache_dir, f"convert-{name}.parquet")
    if not os.path.exists(path):
        dataset.to_parquet(path)
    dataset = spark.read.parquet(path)
    return dataset


def get_transactions(cache_dir):
    dataset = None
    for name in TRANSACTIONS_FILES:
        print(f"Load {name}")
        part = load_dataset("dllllb/alfa-scoring-trx", name, cache_dir=cache_dir)
        key = next(iter(part.keys()))
        part = dataset2spark(part[key], name, cache_dir)
        dataset = dataset.union(part) if dataset is not None else part
    dataset = dataset.selectExpr("app_id as id",
                                 "mcc as labels",
                                 "amnt as amount",
                                 "currency",
                                 "operation_kind",
                                 "operation_type",
                                 "operation_type_group",
                                 "ecommerce_flag",
                                 "payment_system",
                                 "income_flag",
                                 "country",
                                 "city",
                                 "mcc_category",
                                 "card_type",
                                 "transaction_number",
                                 "day_of_week",
                                 "weekofyear",
                                 "hour",
                                 "hour_diff",
                                 "days_before")

    # Convert days_before to timestamps.
    dataset = dataset.withColumn("timestamps", 1000000 - F.col("days_before"))

    # Add log_amount.
    udf = F.udf(lambda x: math.log(abs(x) + 1), FloatType())
    dataset = dataset.withColumn("log_amount", udf(F.col("amount")))
    dataset = dataset.withColumn("log_hour_diff", udf(F.col("hour_diff")))
    dataset = dataset.withColumn("log_days_before", udf(F.col("days_before")))
    return dataset


def get_targets(cache_dir):
    print(f"Load targets")
    # The file is broken:
    #   dataset = load_dataset("dllllb/alfa-scoring-trx", TARGET_FILE, cache_dir=cache_dir)
    #   assert len(dataset.keys()) == 1
    #   key = next(iter(dataset.keys()))
    #   dataset = dataset2spark(dataset[key], "targets", cache_dir)
    # Use a workaround instead:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("dllllb/alfa-scoring-trx", TARGET_FILE + ".csv.gz", cache_dir=cache_dir, repo_type="dataset")
    spark = SparkSession.builder.getOrCreate()
    dataset = spark.read.option("header", True).option("inferschema", "true").csv(path)
    # The end of the workaround.
    dataset = dataset.selectExpr("app_id as id",
                                 "product",
                                 "flag as target")
    dataset = dataset.withColumn("target", F.col("target").cast("int"))
    return dataset


def split_targets_train_test(df):
    # Legacy from PyTorch Lifestream.
    ws = Window.partitionBy("product", "target")
    df = df.withColumn("_hash", F.hash(
        F.concat(F.col("id"), F.lit(42))) / 2 ** 32 + 0.5)
    df = df.withColumn("p", F.row_number().over(ws.orderBy("_hash")) / F.count("*").over(ws)).persist()

    df_target_train = df.where(F.col("p") >= TEST_SIZE).drop("_hash", "p")
    df_target_test = df.where(F.col("p") < TEST_SIZE).drop("_hash", "p")
    return df_target_train.persist(), df_target_test.persist()


def make_index(ids):
    spark = SparkSession.builder.getOrCreate()
    index = spark.createDataFrame([[list(ids)]], ["id"])
    index = index.withColumn("id", F.explode("id"))
    return index


def train_val_test_split(transactions, train_targets, test_targets):
    """Select test set from the labeled subset of the dataset."""

    train_ids = list(sorted({row["id"] for row in train_targets.select("id").distinct().collect()}))
    test_ids = list(sorted({row["id"] for row in test_targets.select("id").distinct().collect()}))

    Random(SEED).shuffle(train_ids)
    n_clients_val = int(len(train_ids) * VAL_SIZE)
    val_ids = set(train_ids[-n_clients_val:])
    train_ids = set(train_ids[:-n_clients_val])

    print("TRAIN LABELED IDS", len(train_ids))
    print("VAL IDS", len(val_ids))
    print("TEST IDS", len(test_ids))

    train_ids = make_index(train_ids)
    val_ids = make_index(val_ids)
    test_ids = make_index(test_ids)
    no_train_ids = val_ids.union(test_ids)

    print("Filter")
    testset = transactions.join(test_ids, on="id", how="inner")
    valset = transactions.join(val_ids, on="id", how="inner")
    trainset = transactions.join(no_train_ids, on="id", how="left_anti")  # Both labeled and unlabeled.
    return trainset.persist(), valset.persist(), testset.persist()


def dump_parquet(df, path, n_partitions):
    df.sort(F.col("id")).repartition(n_partitions, "id").write.mode("overwrite").parquet(path)


def main(args):
    cache_dir = os.path.join(args.root, "cache")
    if not os.path.isdir(args.root):
        os.mkdir(args.root)
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    targets = get_targets(cache_dir)
    train_targets, test_targets = split_targets_train_test(targets)

    transactions = get_transactions(cache_dir)

    print("Transform")
    preprocessor = PysparkDataPreprocessor(
        col_id="id",
        col_event_time="timestamps",
        event_time_transformation="none",
        cols_category=["labels",
                       "currency",
	               "operation_kind",
	               "card_type",
	               "operation_type",
                       "operation_type_group",
                       "ecommerce_flag",
                       "payment_system",
                       "income_flag",
                       "country",
                       "city",
                       "mcc_category",
                       "day_of_week",
                       "hour",
                       "weekofyear"],
        category_transformation="frequency"
    )
    transactions = preprocessor.fit_transform(transactions)
    # Normalize timestamps.
    udf = F.udf(lambda x: [v - x[0] for v in x], ArrayType(IntegerType()))
    transactions = transactions.withColumn("timestamps", udf(F.col("timestamps")))
    transactions = transactions.join(targets.select("id", "target"), on="id", how="left").persist()

    print("Split")
    train, val, test = train_val_test_split(transactions, train_targets, test_targets)

    train_path = os.path.join(args.root, "train.parquet")
    val_path = os.path.join(args.root, "val.parquet")
    test_path = os.path.join(args.root, "test.parquet")
    print(f"Dump train with {train.count()} records to {train_path}")
    dump_parquet(train, train_path, n_partitions=64)
    print(f"Dump val with {val.count()} records to {val_path}")
    dump_parquet(val, val_path, n_partitions=16)
    print(f"Dump test with {test.count()} records to {test_path}")
    dump_parquet(test, test_path, n_partitions=16)
    print("OK")


if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    main(parse_args())
