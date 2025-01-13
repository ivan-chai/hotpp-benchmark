import argparse
import math
import os
import pyspark.sql.functions as F
import tarfile
from huggingface_hub import hf_hub_download
from ptls.preprocessing import PysparkDataPreprocessor
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType
from random import Random


SEED = 42


def parse_args():
    parser = argparse.ArgumentParser("Download, prepare and dump dataset to a parquet file.")
    parser.add_argument("--root", help="Dataset root", default="data")
    return parser.parse_args()


def make_from_folds(folds, data_root, targets_root, dst, n_partitions):
    spark = SparkSession.builder.getOrCreate()
    dataset = None
    for fold in folds:
        part = spark.read.parquet(os.path.join(data_root, f"fold={fold}"))
        targets = spark.read.parquet(os.path.join(targets_root, f"fold={fold}"))
        part = part.join(targets, on="client_id", how="inner")
        dataset = dataset.union(part) if dataset is not None else part
    # Compute log_amount.
    udf = F.udf(lambda x: [math.log(abs(v) + 1) for v in x], ArrayType(FloatType(), False))
    dataset = dataset.withColumn("log_amount", udf(F.col("amount")))
    # Scale event_time by 4 days.
    udf = F.udf(lambda x: [v / 345600 for v in x], ArrayType(FloatType(), False))
    dataset = dataset.withColumn("timestamps", udf(F.col("event_time")))
    # Dump.
    dataset.repartition(n_partitions, "client_id").write.mode("overwrite").parquet(dst)


def load_transactions(root, cache_dir):
    # Download and extract.
    hf_hub_download(repo_id="ai-lab/MBD", filename="ptls.tar.gz", repo_type="dataset", local_dir=cache_dir)
    archive = os.path.join(cache_dir, "ptls.tar.gz")
    with tarfile.open(archive, "r:gz") as fp:
        fp.extractall(path=root)

    hf_hub_download(repo_id="ai-lab/MBD", filename="targets.tar.gz", repo_type="dataset", local_dir=cache_dir)
    archive = os.path.join(cache_dir, "targets.tar.gz")
    with tarfile.open(archive, "r:gz") as fp:
        fp.extractall(path=root)

    # Merge folds.
    make_from_folds([0, 1, 2, 3],
                    os.path.join(root, "ptls", "trx"),
                    os.path.join(root, "targets"),
                    os.path.join(root, "train.parquet"),
                    n_partitions=64)
    make_from_folds([4],
                    os.path.join(root, "ptls", "trx"),
                    os.path.join(root, "targets"),
                    os.path.join(root, "valid.parquet"),
                    n_partitions=16)


def main(args):
    cache_dir = os.path.join(args.root, "cache")
    if not os.path.isdir(args.root):
        os.mkdir(args.root)
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    load_transactions(args.root, cache_dir)
    print("OK")


if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    main(parse_args())
