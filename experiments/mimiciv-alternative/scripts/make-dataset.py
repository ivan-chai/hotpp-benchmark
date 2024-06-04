import argparse
import json
import logging
import math
import os
import shutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from random import Random

import pyspark.sql.functions as F
from icdmappings import Mapper
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, LongType, IntegerType, StringType
from ptls.preprocessing import PysparkDataPreprocessor
from ptls.preprocessing.pyspark.frequency_encoder import FrequencyEncoder


logger = logging.getLogger()


EVENTS_FILENAME = "events_df.parquet"
EVENTS_SIMPLE = "events_df_simple.parquet"
MEASUREMENTS_FILENAME = "dynamic_measurements_df.parquet"
MEASUREMENTS_SIMPLE = "dynamic_measurements_df_simple.parquet"


def parse_args():
    parser = argparse.ArgumentParser("Format fields and split train/test.")
    parser.add_argument("--root", help="Dataset root", default="data")
    return parser.parse_args()


def simplify_events(src, dst):
    df = pq.read_table(source=src).select(["subject_id", "event_id", "event_type", "timestamp"])
    df = df.append_column("label", pc.cast(df["event_type"], "str"))
    df = df.drop("event_type")
    pa.parquet.write_table(df, dst)


def simplify_measurements(src, dst):
    df = pq.read_table(source=src, columns=["measurement_id", "event_id", "icd_code"], filters=pc.is_valid(pc.field("icd_code")))
    df = df.append_column("label", pc.cast(df["icd_code"], "str"))
    df = df.drop("icd_code")
    pa.parquet.write_table(df, dst)


def map_codes(df):
    """Convert ICD 9 to ICD 10 where possible."""
    mapper = Mapper()
    def map_icd(name):
        if name == "UNK":
            return name
        src, value = name.split()
        src = src.lower().replace("_", "")
        return "CH-" + mapper.map(value, source=src, target="chapter")
    icd_mapper = F.udf(map_icd, returnType=StringType())
    df = df.withColumn("labels", icd_mapper(F.col("label"))).drop("label")
    return df


def time2unix(df, time_col, time_fmt):
    if time_fmt in HORIZON_UNITS:
        df = df.withColumn(time_col, F.col(time_col) * HORIZON_UNITS[time_fmt])
    else:
        spark = SparkSession.builder.getOrCreate()
        spark.conf.set("spark.sql.session.timeZone", "UTC")
        df = df.withColumn(time_col, F.unix_timestamp(time_col, time_fmt))
    return df


def extract_time(df):
    """Extract day (float)."""
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    df = df.withColumn("timestamps", F.unix_timestamp(F.to_date(F.col("timestamp"))) / 86400).drop("timestamp")
    return df


def extract_label(df):
    spark = SparkSession.builder.getOrCreate()
    df = df.withColumn("labels", F.explode(F.split(F.col("label"), "&"))).drop("label")
    return df


def split_train_val_test(df):
    spark = SparkSession.builder.getOrCreate()

    users = list(sorted({cl[0] for cl in df.select("id").distinct().collect()}))
    Random(0).shuffle(users)

    # split client list
    n_val = int(len(users) * 0.1)
    n_test = int(len(users) * 0.1)
    users_train = users[:-n_val - n_test]
    users_val = users[-n_val - n_test:-n_test]
    users_test = users[-n_test:]

    users_train = spark.createDataFrame([(i,) for i in users_train], ["id"])
    users_val = spark.createDataFrame([(i,) for i in users_val], ["id"])
    users_test = spark.createDataFrame([(i,) for i in users_test], ["id"])

    # split data
    train = df.join(users_train, on="id", how="inner").persist()
    val = df.join(users_val, on="id", how="inner").persist()
    test = df.join(users_test, on="id", how="inner").persist()

    logger.info(f"Train size: {train.count()}")
    logger.info(f"Val size: {val.count()}")
    logger.info(f"Test size: {test.count()}")

    return train, val, test


def put_diagnoses(df_e, df_m):
    df_e_non_diag = df_e.filter(F.col("labels") != "DIAGNOSIS")
    df_e_diag = df_e.filter(F.col("labels") == "DIAGNOSIS").drop("labels")
    df_e_diag = df_e_diag.join(df_m.select("event_id", "labels"), on="event_id", how="left").fillna("UNK", subset=["labels"])
    return df_e_non_diag.union(df_e_diag).filter(F.col("labels") != "UNK")


def postprocess(part):
    part = part.select("id", "labels", "timestamps")
    part = part.filter(F.size(F.col("labels")) > 20)
    return part


def write(df, path, n_partitions=1):
    if not path.endswith(".parquet"):
        raise ValueError("Output must be parquet file.")
    if os.path.exists(path):
        shutil.rmtree(path)
    df.repartition(n_partitions).persist().write.parquet(path)


def main(root):
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    simplify_events(os.path.join(root, EVENTS_FILENAME), os.path.join(root, EVENTS_SIMPLE))
    simplify_measurements(os.path.join(root, MEASUREMENTS_FILENAME), os.path.join(root, MEASUREMENTS_SIMPLE))

    df_e = spark.read.parquet(os.path.join(root, EVENTS_SIMPLE)).select("subject_id", "event_id", "timestamp", "label")
    df_e = extract_time(df_e)
    df_e = extract_label(df_e)
    df_e = df_e.withColumnRenamed("subject_id", "id")
    df_e = df_e.persist()

    df_m = spark.read.parquet(os.path.join(root, MEASUREMENTS_SIMPLE)).select("event_id", "label")
    df_m = map_codes(df_m).persist()

    df = put_diagnoses(df_e, df_m).persist()

    df_train, df_val, df_test = split_train_val_test(df)

    preprocessor = PysparkDataPreprocessor(
        col_id="id",
        col_event_time="event_id",
        event_time_transformation="none",
        cols_category=["labels"],
        cols_identity=["timestamps"]
    )
    df_train = postprocess(preprocessor.fit_transform(df_train))
    df_val = postprocess(preprocessor.transform(df_val))
    df_test = postprocess(preprocessor.transform(df_test))

    logger.info(f"Output schema: {df_train.schema}.")
    logger.info(f"Final train size: {df_train.count()}")
    logger.info(f"Final val size: {df_val.count()}")
    logger.info(f"Final test size: {df_test.count()}")

    write(df_train, os.path.join(root, "train.parquet"), n_partitions=32)
    write(df_val, os.path.join(root, "val.parquet"))
    write(df_test, os.path.join(root, "test.parquet"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_args().root)
    logger.info("Done")
