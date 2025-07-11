import argparse
import math
import os
import scipy.io
import pandas as pd
import pyspark.sql.functions as F
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType, ArrayType
from random import Random


SEED = 42
VAL_SIZE = 0.1


def parse_args():
    parser = argparse.ArgumentParser("Prepare and dump dataset to a parquet file.")
    parser.add_argument("--root", help="Dataset root", default="data")
    parser.add_argument("--chunked", action="store_true", help="Make a chunked version of the dataset.")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--chunk-stride", type=int, default=1)
    return parser.parse_args()


def read_arff(path, id_offset=0, chunked=False, chunk_size=16, chunk_stride=8):
    spark = SparkSession.builder.getOrCreate()
    df = pd.DataFrame(scipy.io.arff.loadarff(path)[0])
    df["id"] = df.reset_index().index + id_offset
    df = spark.createDataFrame(df)
    features = [f"t-{i}" for i in range(1, 61)]
    class_udf = F.udf(lambda x: int(x), IntegerType())
    df = df.select(
        "id",
        F.array(*[F.col(name) for name in features]).alias("bins"),
        class_udf(F.col("class")).alias("target")
    )
    length = 60
    if chunked:
        assert chunk_size <= length
        def make_chunks(x):
            assert len(x) == length
            offset = (length - chunk_size) % chunk_stride
            x = x[offset:]
            x = [x[i: i + chunk_size]
                 for i in range(0, len(x) - chunk_size + 1, chunk_stride)]  # (N, S).
            return x
        df = df.withColumn("bins", F.udf(make_chunks, ArrayType(ArrayType(FloatType(), False), False))(F.col("bins")))
        length = int(math.ceil((length - chunk_size + 1) / chunk_stride))
    return df.withColumn("timestamps", F.lit(list(map(float, range(length)))))


def train_val_split(transactions):
    """Select test set from the labeled subset of the dataset."""
    data_ids = {row["id"] for row in transactions.select("id").distinct().collect()}
    data_ids = list(sorted(data_ids))

    Random(SEED).shuffle(data_ids)
    n_clients_val = int(len(data_ids) * VAL_SIZE)
    val_ids = set(data_ids[:n_clients_val])
    train_ids = set(data_ids[n_clients_val:])

    trainset = transactions.filter(transactions["id"].isin(train_ids))
    valset = transactions.filter(transactions["id"].isin(val_ids))
    return trainset.persist(), valset.persist()


def dump_parquet(df, path, n_partitions):
    df.sort(F.col("id")).repartition(n_partitions, "id").write.mode("overwrite").parquet(path)


def main(args):
    dev = read_arff(os.path.join(args.root, "SharePriceIncrease_TRAIN.arff"),
                    chunked=args.chunked, chunk_size=args.chunk_size, chunk_stride=args.chunk_stride)
    test = read_arff(os.path.join(args.root, "SharePriceIncrease_TEST.arff"),
                     id_offset=dev.count(),
                     chunked=args.chunked, chunk_size=args.chunk_size, chunk_stride=args.chunk_stride)
    train, val = train_val_split(dev)

    # Dump.
    suffix = "-chunked" if args.chunked else ""
    train_path = os.path.join(args.root, f"train{suffix}.parquet")
    val_path = os.path.join(args.root, f"val{suffix}.parquet")
    test_path = os.path.join(args.root, f"test{suffix}.parquet")
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
