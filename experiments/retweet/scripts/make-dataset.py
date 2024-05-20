import argparse
import numpy as np
import os
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


TIME_SCALE = 3600  # 1 hour.

SMALL_THRESHOLD = 120
MEDIUM_THRESHOLD = 1363


def parse_args():
    parser = argparse.ArgumentParser("Prepare and dump dataset to a parquet file.")
    parser.add_argument("--root", help="Dataset root", default="data")
    return parser.parse_args()


def dump_parquet(data, path):
    # Timestamps are integer values.
    ids = data["seq_idx"]
    timestamps = [np.array(v, dtype=int) for v in data["time_since_start"]]
    labels = [np.array(v, dtype=int) for v in data["type_event"]]
    table = pa.table({"id": ids, "timestamps": timestamps, "labels": labels})
    pa.parquet.write_table(table, path)


def read_data(path):
    with open(path) as fp:
        header = fp.readline().strip()
        assert header == "relative_time_second,number_of_followers"
    data = np.loadtxt(path, delimiter=",", dtype=float, skiprows=1).astype(int)
    print(f"Loaded data file with shape {data.shape}")
    times = data[:, 0] / TIME_SCALE
    labels = np.zeros([len(data)], dtype=int)
    labels[data[:, 1] >= SMALL_THRESHOLD] = 1
    labels[data[:, 1] >= MEDIUM_THRESHOLD] = 2
    return times, labels


def read_index(path):
    with open(path) as fp:
        header = fp.readline().strip()
        assert header == "tweet_id,post_time_day,start_ind,end_ind"
    data = np.loadtxt(path, delimiter=",", dtype=float, skiprows=1)
    print(f"Loaded index with shape {data.shape}")
    dates = data[:, 1]  # Float offset in days.
    index = data[:, 2:4].astype(int)  # Start and end indices.
    index[:, 0] -= 1  # Now each range is [start, stop).
    return dates, index


def split(ids, index_dates):
    # There is a total of 15 days.
    # Train / dev / test are split by the date of the original tweet.
    # Train: 7 days, dev: 4 days, test: 4 days.
    mask = index_dates < 7
    train_ids = ids[mask]
    mask = np.logical_and(index_dates >= 7, index_dates < 11)
    dev_ids = ids[mask]
    mask = index_dates >= 11
    test_ids = ids[mask]
    return train_ids, dev_ids, test_ids


def make_and_dump(ids, index, times, labels, path, n_partitions=1):
    time_seqs = []
    label_seqs = []
    for start, stop in index[ids]:
        s_times = times[start:stop]
        s_labels = labels[start:stop]
        order = np.argsort(s_times)
        s_times = s_times[order]
        s_labels = s_labels[order]
        time_seqs.append(s_times.tolist())
        label_seqs.append(s_labels.tolist())
    ids = pa.array(ids)
    time_seqs = pa.array(time_seqs)
    label_seqs = pa.array(label_seqs)
    table = pa.Table.from_arrays([ids, time_seqs, label_seqs], ["id", "timestamps", "labels"])

    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith(".parquet"):
                os.remove(os.path.join(path, filename))
    else:
        os.mkdir(path)

    for i in range(n_partitions):
        part_path = os.path.join(path, f"part-{i:05d}.parquet")
        pq.write_table(table.filter(pc.field("id") - pc.floor(pc.field("id") / n_partitions) * n_partitions == i), part_path)


def main(args):
    times, labels = read_data(os.path.join(args.root, "data.csv"))
    index_dates, index = read_index(os.path.join(args.root, "index.csv"))

    ids = np.arange(len(index_dates))
    train_ids, dev_ids, test_ids = split(ids, index_dates)

    make_and_dump(test_ids, index, times, labels, os.path.join(args.root, "test.parquet"))
    make_and_dump(dev_ids, index, times, labels, os.path.join(args.root, "dev.parquet"))
    make_and_dump(train_ids, index, times, labels, os.path.join(args.root, "train.parquet"),
                  n_partitions=32)


if __name__ == "__main__":
    main(parse_args())
