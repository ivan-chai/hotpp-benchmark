from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]

DATASETS = ["stackoverflow", "amazon", "retweet", "mimiciv"]
TIME_UNITS = {
    "stackoverflow": "days",
    "amazon": "days",
    "retweet": "seconds",
    "mimiciv": "hours",
}
N_SEQS = 6
SEED = 7


def load(ds):
    return pd.read_parquet(ROOT / f"experiments/{ds}/data/train.parquet")


def build_palette(n_classes):
    if n_classes <= 10:
        base = plt.cm.tab10.colors
    elif n_classes <= 20:
        base = plt.cm.tab20.colors
    else:
        base = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
    colors = [base[i % len(base)] for i in range(max(n_classes, 1))]
    return ListedColormap(colors)


def pick_samples(df, k):
    long_enough = df[df["timestamps"].apply(len) >= 8]
    if len(long_enough) < k:
        long_enough = df
    return long_enough.sample(min(k, len(long_enough)), random_state=SEED)


def plot_row(ax, ds, df):
    samples = pick_samples(df, N_SEQS)

    used_classes = sorted(
        {int(c) for row in samples["labels"] for c in np.asarray(row)}
    )
    cmap = build_palette(len(used_classes))
    class_to_idx = {c: i for i, c in enumerate(used_classes)}

    units = TIME_UNITS[ds]
    for i, (_, row) in enumerate(samples.iterrows()):
        ts = np.asarray(row["timestamps"], dtype=float)
        ls = np.asarray(row["labels"])
        ts = ts - ts[0]
        color_idx = np.array([class_to_idx[int(c)] for c in ls])
        ax.scatter(
            ts,
            np.full_like(ts, i, dtype=float),
            c=color_idx,
            cmap=cmap,
            vmin=-0.5,
            vmax=len(used_classes) - 0.5,
            s=120,
            marker="|",
            linewidths=2.0,
        )
        ax.scatter(
            ts,
            np.full_like(ts, i, dtype=float),
            c=color_idx,
            cmap=cmap,
            vmin=-0.5,
            vmax=len(used_classes) - 0.5,
            s=55,
            edgecolor="black",
            linewidths=0.4,
            alpha=0.95,
        )

    ax.set_yticks(list(range(len(samples))))
    ax.set_yticklabels([f"#{i+1}" for i in range(len(samples))])
    ax.set_ylim(-0.7, len(samples) - 0.3)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlabel(f"time since start  ({units})")

    n_classes_total = int(np.max(np.concatenate([np.asarray(l) for l in df["labels"].head(20000)]))) + 1
    ax.set_title(
        f"{ds}   (N={len(df)} sequences, {n_classes_total} classes)",
        fontweight="bold",
    )


def main():
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(13, 2.4 * len(DATASETS)))
    for ax, ds in zip(axes, DATASETS):
        df = load(ds)
        plot_row(ax, ds, df)
    fig.suptitle("HoTPP datasets — example event sequences", fontsize=15, fontweight="bold")
    fig.tight_layout()
    out = ROOT / "scripts" / "dataset_examples.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
