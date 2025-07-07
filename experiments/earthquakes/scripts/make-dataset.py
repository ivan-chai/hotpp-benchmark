import argparse
import os
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType
from random import Random


SEED = 42
VAL_SIZE = 0.1


def parse_args():
    parser = argparse.ArgumentParser("Prepare and dump dataset to a parquet file.")
    parser.add_argument("--root", help="Dataset root (must contain MIMIC4 folder with `core`, `hosp`, and `icu`)", default="data")
    return parser.parse_args()


def read_txt(path, id_offset=0):
    spark = SparkSession.builder.getOrCreate()
    df = pd.read_table(path, header=None, sep=r"\s+", skipinitialspace=True)
    df["id"] = df.reset_index().index + id_offset
    df = spark.createDataFrame(df)
    features = [str(i) for i in range(1, 513)]
    df = df.select(
        "id",
        F.lit(list(map(float, range(0, 512)))).alias("timestamps"),
        F.array(*[F.col(name) for name in features]).alias("values"),
        F.col("0").cast("int").alias("target")
    )
    return df


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
    dev = read_txt(os.path.join(args.root, "Earthquakes_TRAIN.txt"))
    test = read_txt(os.path.join(args.root, "Earthquakes_TEST.txt"),
                    id_offset=dev.count())
    train, val = train_val_split(dev)

    # Dump.
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
