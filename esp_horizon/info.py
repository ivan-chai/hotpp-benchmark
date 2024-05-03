import hydra
import torch


def get_horizon_lengths(dataset, horizon):
    lengths = []
    field = dataset.timestamps_field
    for features in dataset:
        ts = features[field]
        indices = torch.arange(len(ts))
        end_indices = torch.searchsorted(ts, ts + horizon, side="left")
        lengths.append(end_indices - indices)
    return torch.cat(lengths)


@hydra.main(version_base="1.2", config_path=None)
def main(conf):
    model = hydra.utils.instantiate(conf.module)
    dm = hydra.utils.instantiate(conf.data_module)
    print("======== MODEL ========")
    print(model)
    print("======== DATASET ========")
    for split in ["train", "dev", "test"]:
        if not hasattr(dm, f"{split}_data"):
            continue
        dataset = getattr(dm, f"{split}_data")
        print(f"SPLIT {split}")
        print(f"  Size: {len(dataset)}")
        field = dataset.timestamps_field
        seq_lens = torch.tensor([len(v[field]) for v in dataset])
        print(f"  Min seq. length: {seq_lens.min().item()}")
        print(f"  Max seq. length: {seq_lens.max().item()}")
        print(f"  Avg seq. length: {seq_lens.float().mean().item()}")
        try:
            horizon = getattr(model, f"_{split}_metric").horizon
            hor_lens = get_horizon_lengths(dataset, horizon)
            print(f"  Min horizon length: {hor_lens.min().item()}")
            print(f"  Max horizon length: {hor_lens.max().item()}")
            print(f"  Avg horizon length: {hor_lens.float().mean().item()}")
        except AttributeError:
            pass


if __name__ == "__main__":
    main()
