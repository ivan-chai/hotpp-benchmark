
from __future__ import annotations

import os
os.environ.setdefault("WANDB_MODE", "offline")

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


ROOT = "/root/.nv/hotpp-benchmark"
DATASETS = [
    ("stackoverflow", f"{ROOT}/experiments/stackoverflow/configs",
     f"{ROOT}/experiments/stackoverflow/checkpoints/next_item_mamba.ckpt"),
    ("amazon",        f"{ROOT}/experiments/amazon/configs",
     f"{ROOT}/experiments/amazon/checkpoints/next_item_mamba.ckpt"),
    ("retweet",       f"{ROOT}/experiments/retweet/configs",
     f"{ROOT}/experiments/retweet/checkpoints/next_item_mamba.ckpt"),
    ("mimiciv",       f"{ROOT}/experiments/mimiciv/configs",
     f"{ROOT}/experiments/mimiciv/checkpoints/next_item_mamba.ckpt"),
]

MAX_BATCHES = 6  # limit per dataset to avoid gigantic arrays
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collect_mamba_dt(config_dir, ckpt_path):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath(config_dir)):
        cfg = compose(config_name="next_item_mamba")
    os.chdir(os.path.dirname(os.path.abspath(config_dir)))

    dm = hydra.utils.instantiate(cfg.data_module)
    dm.setup("test")
    module = hydra.utils.instantiate(cfg.module)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    module.load_state_dict(sd, strict=False)
    module.eval().to(DEVICE)

    mamba = module._seq_encoder.model.model

    captured = []
    hooks = []

    def make_hook(layer_idx):
        def _hook(mod, inp, out):
            # out: pre-softplus dt, shape (B, L, intermediate_size)
            dt = F.softplus(out).detach().float().cpu()
            captured.append((layer_idx, dt))
        return _hook

    for i, layer in enumerate(mamba.layers):
        hooks.append(layer.mixer.dt_proj.register_forward_hook(make_hook(i)))

    dts_per_layer = {}
    reals = []
    try:
        dl = dm.test_dataloader(rank=0, world_size=1)
        with torch.no_grad():
            for bi, batch in enumerate(dl):
                if bi >= MAX_BATCHES:
                    break
                x, _ = batch
                x = x.to(DEVICE)
                captured.clear()
                module(x, return_states=False)

                ts = x.payload["timestamps"]
                lens = x.seq_lens
                for j in range(ts.shape[0]):
                    n = int(lens[j].item())
                    if n >= 2:
                        reals.append((ts[j, 1:n] - ts[j, :n - 1]).float().cpu().numpy())

                for (li, dt) in captured:
                    mask = torch.zeros(dt.shape[0], dt.shape[1], dtype=torch.bool)
                    for j in range(dt.shape[0]):
                        n = int(lens[j].item())
                        mask[j, :n] = True
                    dts_per_layer.setdefault(li, []).append(dt[mask].flatten().numpy())
    finally:
        for h in hooks:
            h.remove()

    per_layer = {li: np.concatenate(arrs) for li, arrs in dts_per_layer.items()}
    real = np.concatenate(reals) if reals else np.array([])
    real = real[np.isfinite(real) & (real >= 0)]
    return per_layer, real


def main():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for ax, (name, cfg_dir, ckpt) in zip(axes, DATASETS):
        per_layer, real = collect_mamba_dt(cfg_dir, ckpt)
        delta = per_layer[0]
        delta = delta[delta > 0]
        real = real[real > 0]

        print(
            f"{name}: Δ(layer0): mean={delta.mean():.4g} median={np.median(delta):.4g} "
            f"q01={np.quantile(delta, 0.01):.4g} q99={np.quantile(delta, 0.99):.4g} | "
            f"real dt: mean={real.mean():.4g} median={np.median(real):.4g} "
            f"q01={np.quantile(real, 0.01):.4g} q99={np.quantile(real, 0.99):.4g}"
        )

        lo = min(np.quantile(delta, 0.001), np.quantile(real, 0.001))
        hi = max(np.quantile(delta, 0.999), np.quantile(real, 0.999))
        bins = np.logspace(np.log10(lo), np.log10(hi), 80)

        w_d = np.full_like(delta, 1.0 / len(delta))
        w_r = np.full_like(real, 1.0 / len(real))
        ax.hist(delta, bins=bins, weights=w_d, alpha=0.55, color="#d62728",
                label=f"Mamba Δ (layer 0, mean={delta.mean():.3g}, median={np.median(delta):.3g})")
        ax.hist(real, bins=bins, weights=w_r, alpha=0.45, color="#1f77b4",
                label=f"real dt (mean={real.mean():.3g}, median={np.median(real):.3g})")
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_xlabel("value (log scale)")
        ax.set_ylabel("fraction of samples")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.25)

    plt.suptitle("Internal Mamba discretization step Δ = softplus(dt_proj(x)) vs real inter-event dt "
                 "(layer 0, log-x)", fontsize=13)
    plt.tight_layout()
    out = f"{ROOT}/scripts/all_mamba_internal_dt.png"
    plt.savefig(out, dpi=130)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
