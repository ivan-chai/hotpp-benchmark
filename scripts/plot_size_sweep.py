from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from transformers import MambaConfig, MambaModel


ROOT = Path("/root/.nv/hotpp-benchmark")
DATASETS = ["amazon", "retweet", "stackoverflow", "mimiciv"]

PAT = re.compile(r"d(?P<d>\d+)_L(?P<l>\d+)\.yaml$")


def load_vocab(dataset):
    with open(ROOT / "experiments" / dataset / "configs" / "default.yaml") as f:
        return int(yaml.safe_load(f)["num_classes"])


def count_params(hidden, layers, vocab):
    cfg = MambaConfig(
        hidden_size=hidden,
        num_hidden_layers=layers,
        vocab_size=vocab,
    )
    m = MambaModel(cfg)
    n = sum(p.numel() for p in m.parameters())
    del m
    return n


def collect(dataset):
    results_dir = ROOT / "experiments" / dataset / "results" / "mamba_sizes"
    if not results_dir.is_dir():
        return []
    vocab = load_vocab(dataset)
    out = []
    for p in sorted(results_dir.glob("d*_L*.yaml")):
        m = PAT.search(p.name)
        if not m:
            continue
        d, l = int(m["d"]), int(m["l"])
        with open(p) as f:
            rep = yaml.safe_load(f) or {}
        rec = {
            "tag": f"d{d}_L{l}",
            "d": d,
            "l": l,
            "n_params": count_params(d, l, vocab),
            "test_tmap": rep.get("test/T-mAP"),
            "val_tmap": rep.get("val/T-mAP"),
            "test_tmap_w": rep.get("test/T-mAP-weighted"),
            "val_tmap_w": rep.get("val/T-mAP-weighted"),
        }
        out.append(rec)
    return out


def main():
    data = {ds: collect(ds) for ds in DATASETS}
    data = {k: v for k, v in data.items() if v}
    n = len(data)
    if n == 0:
        return

    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    l_colors = {1: "#1f77b4", 2: "#2ca02c", 4: "#d62728", 8: "#9467bd"}

    for ax, (ds, recs) in zip(axes_flat, data.items()):
        recs = sorted(recs, key=lambda r: r["n_params"])
        xs = [r["n_params"] for r in recs]
        ys_test = [r["test_tmap"] for r in recs]
        ys_val = [r["val_tmap"] for r in recs]

        ax.plot(xs, ys_test, "-", color="#444", alpha=0.35, zorder=1)
        for r in recs:
            c = l_colors.get(r["l"], "#555")
            ax.scatter(r["n_params"], r["test_tmap"], color=c, s=70, zorder=3,
                       edgecolor="black", linewidth=0.5)
            ax.annotate(r["tag"], (r["n_params"], r["test_tmap"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.plot(xs, ys_val, "--", color="#aaa", alpha=0.7, label="val/T-mAP", zorder=2)

        ax.set_xscale("log")
        ax.set_title(ds)
        ax.set_xlabel("# parameters")
        ax.set_ylabel("T-mAP")
        ax.grid(True, which="both", alpha=0.25)

        best = max(recs, key=lambda r: r["test_tmap"] or -1)
        ax.axhline(best["test_tmap"], color="#999", linestyle=":", alpha=0.6)
        handles = [plt.Line2D([], [], marker="o", linestyle="", color=c,
                              markeredgecolor="black", markersize=8, label=f"L={l}")
                   for l, c in l_colors.items() if any(r["l"] == l for r in recs)]
        handles.append(plt.Line2D([], [], color="#aaa", linestyle="--", label="val/T-mAP"))
        ax.legend(handles=handles, fontsize=8, loc="best")

    for ax in axes_flat[len(data):]:
        ax.axis("off")

    plt.suptitle("Plain Mamba: T-mAP vs model size", fontsize=14)
    plt.tight_layout()
    out = ROOT / "scripts" / "size_sweep_tmap.png"
    plt.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
