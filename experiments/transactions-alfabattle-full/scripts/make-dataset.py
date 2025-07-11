import argparse
import math
import os
import pathlib
import pyspark.sql.functions as F
import shutil
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
TEST_SIZE = 0.1
VAL_SIZE = 0.05


def parse_args():
    parser = argparse.ArgumentParser("Download, prepare and dump dataset to a parquet file.")
    parser.add_argument("--root", help="Dataset root", default="data")
    return parser.parse_args()


def download(cache_dir):
    spark = SparkSession.builder.getOrCreate()
    dataset = None
    for name in TRANSACTIONS_FILES:
        path = os.path.join(cache_dir, f"convert-{name}.parquet")
        if not os.path.exists(path):
            print(f"Load {name}")
            part = load_dataset("dllllb/alfa-scoring-trx", name, cache_dir=cache_dir)
            key = next(iter(part.keys()))
            part[key].to_parquet(path)
        part = spark.read.parquet(path).select(
            "app_id", "amnt", "currency", "operation_kind", "card_type",
            "operation_type", "operation_type_group", "ecommerce_flag",
            "payment_system", "income_flag", "mcc", "country", "city",
            "mcc_category", "day_of_week", "hour", "days_before", "weekofyear",
            "hour_diff", "transaction_number"
        )
        dataset = part if dataset is None else dataset.union(part)
    split_path = os.path.join(cache_dir, f"split.parquet")
    if not os.path.exists(split_path):
        dataset.repartition(100, "app_id").write.mode("overwrite").parquet(split_path)


def get_targets(cache_dir):
    print(f"Load targets")
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


def read_part(path):
    spark = SparkSession.builder.getOrCreate()
    dataset = spark.read.parquet(path)
    dataset = dataset.selectExpr("app_id as id",
                                 "mcc as labels",
                                 "amnt as log_amount",
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

    # Add log deltas.
    udf = F.udf(lambda x: math.log(abs(x) + 1), FloatType())
    dataset = dataset.withColumn("log_hour_diff", udf(F.col("hour_diff")))
    dataset = dataset.withColumn("log_days_before", udf(F.col("days_before")))
    return dataset


def collect_lists(df, group_field="id", sort_field="transaction_number"):
    col_list = [sort_field] + [col for col in df.columns if (col != group_field) and (col != sort_field)]
    unpack_col_list = [group_field] + [F.col(f"_struct.{col}").alias(col) for col in col_list]

    df = df.groupBy(group_field).agg(F.sort_array(F.collect_list(F.struct(*col_list))).alias("_struct"))
    df = df.select(*unpack_col_list).drop("_struct").persist()

    # Measure time from the beginning.
    udf = F.udf(lambda x: [v - x[0] for v in x], ArrayType(IntegerType()))
    df = df.withColumn("timestamps", udf(F.col("timestamps")))
    return df


def dump_parquet(df, path, n_partitions):
    df.sort(F.col("id")).repartition(n_partitions, "id").write.mode("overwrite").parquet(path)


def dump_parquet_file(df, path):
    root = "folder-" + path
    df.sort(F.col("id")).repartition(1, "id").write.mode("overwrite").parquet(root)
    parquet_files = list(pathlib.Path(root).glob("*.parquet"))
    assert len(parquet_files) == 1
    shutil.move(parquet_files[0], path)
    shutil.rmtree(root)


def convert_part(src_path,
                 dst_train, dst_val, dst_test,
                 targets_train, targets_test):
    df = read_part(src_path)
    df = collect_lists(df)

    df_dev = df.join(targets_test, on="id", how="left_anti").join(targets_train, on="id", how="left")
    dump_parquet_file(df_dev.filter(F.col("id") % 100 >= VAL_SIZE * 100), dst_train)
    dump_parquet_file(df_dev.filter(F.col("id") % 100 < VAL_SIZE * 100), dst_val)

    df_test = df.join(targets_test, on="id", how="inner")
    dump_parquet_file(df_test, dst_test)


def clean_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def main(args):
    cache_dir = os.path.join(args.root, "cache")
    if not os.path.isdir(args.root):
        os.mkdir(args.root)
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    download(cache_dir)
    targets = get_targets(cache_dir)
    targets_train, targets_test = split_targets_train_test(targets)

    root_dst_train = os.path.join(args.root, "train.parquet")
    root_dst_val = os.path.join(args.root, "val.parquet")
    root_dst_test = os.path.join(args.root, "test.parquet")
    clean_folder(root_dst_train)
    clean_folder(root_dst_val)
    clean_folder(root_dst_test)
    root_src = os.path.join(cache_dir, f"split.parquet")
    for filename in pathlib.Path(root_src).glob("*.parquet"):
        filename = os.path.basename(filename)
        convert_part(os.path.join(root_src, filename),
                     os.path.join(root_dst_train, filename),
                     os.path.join(root_dst_val, filename),
                     os.path.join(root_dst_test, filename),
                     targets_train, targets_test)
    print("OK")


if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    main(parse_args())
