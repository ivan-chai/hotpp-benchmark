import argparse
import math
import os
import pyspark.sql.functions as F
from datasets import load_dataset
from ptls.preprocessing import PysparkDataPreprocessor
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import FloatType
from random import Random


SEED = 42
VAL_SIZE = 0.0  # 0.1
TEST_SIZE = 0.2


def parse_args():
    parser = argparse.ArgumentParser("Prepare and dump dataset to a parquet file.")
    parser.add_argument("--root", help="Dataset root (must contain MIMIC4 folder with `core`, `hosp`, and `icu`)", default="data")
    return parser.parse_args()


def load_admissions(root):
    spark = SparkSession.builder.getOrCreate()

    path = os.path.join(root, "MIMIC4/core/admissions.csv")
    admissions = spark.read.option("delimiter", ",").option("header", True).csv(path).select("subject_id", "hadm_id", "admittime", "deathtime")
    admissions = admissions.withColumn("target", F.col("deathtime").isNotNull()).drop("deathtime")
    admissions = admissions.selectExpr("subject_id as id",
                                       "hadm_id",
                                       "admittime as timestamps",
                                       "target")
    return admissions.coalesce(1).cache()


def get_targets(admissions):
    targets = admissions.select("id", "target").dropna().groupBy("id").agg((F.sum(F.col("target").cast("int")) > 0).alias("target"))
    return targets.coalesce(1).cache()


def load_diagnoses(root, admissions):
    spark = SparkSession.builder.getOrCreate()

    path = os.path.join(root, "MIMIC4/hosp/diagnoses_icd.csv")
    df = spark.read.option("delimiter", ",").option("header", True).csv(path).selectExpr(
        "subject_id as id",
        "hadm_id",
        "seq_num",
        "icd_code as labels")
    #   "icd_version")
    df = df.dropna()
    # TODO: Unify ICD 9 and 10.

    df = df.join(admissions.select("id", "hadm_id", "timestamps").dropna(), on=["id", "hadm_id"]).drop("hadm_id")
    df = df.withColumn("types", F.lit("diagnosis"))
    return df.select("id", "timestamps", "seq_num", "labels", "types")


def load_procedures(root, admissions):
    spark = SparkSession.builder.getOrCreate()

    path = os.path.join(root, "MIMIC4/hosp/procedures_icd.csv")
    df = spark.read.option("delimiter", ",").option("header", True).csv(path).selectExpr(
        "subject_id as id",
        "hadm_id",
        "seq_num",
        "icd_code as labels")
    #   "icd_version")
    df = df.dropna()
    # TODO: Unify ICD 9 and 10.

    df = df.join(admissions.select("id", "hadm_id", "timestamps").dropna(), on=["id", "hadm_id"]).drop("hadm_id")
    df = df.withColumn("types", F.lit("procedure"))
    return df.select("id", "timestamps", "seq_num", "labels", "types")


def load_prescriptions(root):
    spark = SparkSession.builder.getOrCreate()

    path = os.path.join(root, "MIMIC4/hosp/prescriptions.csv")
    df = spark.read.option("delimiter", ",").option("header", True).csv(path).selectExpr(
        "subject_id as id",
        "starttime as timestamps",
        "drug as labels")
    #   "icd_version")
    df = df.dropna()
    # TODO: Unify ICD 9 and 10.

    df = df.withColumn("seq_num", F.lit(1)).withColumn("types", F.lit("prescription"))
    return df.select("id", "timestamps", "seq_num", "labels", "types")


def load_labs(root):
    spark = SparkSession.builder.getOrCreate()

    path = os.path.join(root, "MIMIC4/hosp/labevents.csv")
    df = spark.read.option("delimiter", ",").option("header", True).csv(path).selectExpr(
        "subject_id as id",
        "labevent_id as seq_num",
        "charttime as timestamps",
        "itemid as labels")
    #   "icd_version")
    df = df.dropna()
    # TODO: Unify ICD 9 and 10.

    df = df.withColumn("types", F.lit("lab"))
    return df.select("id", "timestamps", "seq_num", "labels", "types")


#def train_val_test_split(transactions, targets):
#    """Select test set from the labeled subset of the dataset."""
#    data_ids = {row["id"] for row in transactions.select("id").distinct().collect()}
#    labeled_ids = {row["id"] for row in targets.select("id").distinct().collect()}
#    labeled_ids = data_ids & labeled_ids
#    unlabeled_ids = data_ids - labeled_ids
#
#    labeled_ids = list(sorted(list(labeled_ids)))
#    Random(SEED).shuffle(labeled_ids)
#    n_clients_test = int(len(labeled_ids) * TEST_SIZE)
#    test_ids = set(labeled_ids[-n_clients_test:])
#    train_ids = list(sorted(set(labeled_ids[:-n_clients_test]) | unlabeled_ids))
#    Random(SEED + 1).shuffle(train_ids)
#    n_clients_val = int(len(train_ids) * VAL_SIZE)
#    val_ids = set(train_ids[-n_clients_val:])
#    train_ids = set(train_ids[:-n_clients_val])
#
#    testset = transactions.filter(transactions["id"].isin(test_ids))
#    trainset = transactions.filter(transactions["id"].isin(train_ids))
#    valset = transactions.filter(transactions["id"].isin(val_ids))
#    return trainset.persist(), valset.persist(), testset.persist()

# Compatibility with Makarenko.
def train_val_test_split(transactions, targets):
    """Select test set from the labeled subset of the dataset."""
    spark = SparkSession.builder.getOrCreate()

    labeled_ids = {row["id"] for row in targets.select("id").distinct().collect()}
    data_ids = [row["id"] for row in transactions.select("id").distinct().collect()]
    data_ids = list(sorted(data_ids))
    test_size = int(len(data_ids) * TEST_SIZE)
    train_ids = data_ids[:-test_size]
    test_ids = list(set(data_ids[-test_size:]) & labeled_ids)

    train_index = spark.createDataFrame([(a,) for a in train_ids], ["id"])
    test_index = spark.createDataFrame([(a,) for a in test_ids], ["id"])
    val_index = test_index.limit(0)

    testset = transactions.join(test_index, on="id")
    valset = transactions.join(val_index, on="id")
    trainset = transactions.join(train_index, on="id")
    return trainset.persist(), valset.persist(), testset.persist()


def dump_parquet(df, path, n_partitions):
    df.sort(F.col("id")).repartition(n_partitions, "id").write.mode("overwrite").parquet(path)


def main(args):
    admissions = load_admissions(args.root)
    diagnoses = load_diagnoses(args.root, admissions)
    procedures = load_procedures(args.root, admissions)
    prescriptions = load_prescriptions(args.root)
    labs = load_labs(args.root)

    # Join.
    transactions = diagnoses.union(procedures).union(prescriptions).union(labs)  # id, timestamps, seq_num, labels, types.
    # Preprocess
    transactions = transactions.withColumn("id", F.col("id").cast("long"))
    transactions = transactions.withColumn("seq_num", F.col("seq_num").cast("long"))
    transactions = transactions.withColumn("labels", F.trim(F.col("labels")))
    transactions = transactions.withColumn("timestamps", F.unix_timestamp(F.col("timestamps")))
    min_timestamp = int(transactions.agg({"timestamps": "min"}).collect()[0]["min(timestamps)"])
    print("Minimum timestamp", min_timestamp)
    transactions = transactions.withColumn("timestamps", (F.col("timestamps") - min_timestamp).cast("float") / (60 * 60 * 24))  # Days.

    # Group.
    transactions = transactions.withColumn("order", F.struct("timestamps", "seq_num")).drop("seq_num")  # id, order, timestamps, labels, types.
    targets = get_targets(admissions)  # id, target.

    print("Transform")
    preprocessor = PysparkDataPreprocessor(
        col_id="id",
        col_event_time="order",
        event_time_transformation="none",
        cols_category=["labels", "types"],
        category_transformation="frequency",
        cols_identity=["timestamps"]
    )
    transactions = preprocessor.fit_transform(transactions).drop("order", "event_time")  # id, timestamps, labels, types.
    transactions = transactions.join(targets, on="id", how="left").persist()
    print("N clients:", transactions.count())

    print("Split")
    # Lifestream compatibility split.
    transactions = transactions.withColumn("id", F.col("id").cast("string"))
    targets = targets.withColumn("id", F.col("id").cast("string"))
    # Split.
    train, val, test = train_val_test_split(transactions, targets)
    # Revert compatibility cast.
    train = train.withColumn("id", F.col("id").cast("int"))
    val = val.withColumn("id", F.col("id").cast("int"))
    test = test.withColumn("id", F.col("id").cast("int"))

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
