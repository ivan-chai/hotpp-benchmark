import hydra
import torch


def get_horizon_lengths(dataset, horizon):
    lengths = []
    field = dataset.timestamps_field
    for features in dataset:
        ts = features[field]
        indices = torch.arange(len(ts))
        end_indices = torch.searchsorted(ts, ts + horizon, side="left")
        lengths.append((end_indices - indices - 1).clip(min=0))
    return torch.cat(lengths)


class Metric:
    def __init__(self):
        self._min = None
        self._max = None
        self._avgs = []
        self._medians = []

    def update(self, x):
        x = torch.as_tensor(x)
        v = x.min().item()
        self._min = min(self._min, v) if self._min is not None else v
        v = x.max().item()
        self._max = max(self._max, v) if self._max is not None else v
        self._avgs.append(x.float().mean().item())
        self._medians.append(x.float().median().item())

    def compute(self):
        return {
            "min": self._min,
            "max": self._max,
            "avg": torch.tensor(self._avgs).mean().item(),
            "median": torch.tensor(self._medians).median().item()
        }


@hydra.main(version_base="1.2", config_path=None)
def main(conf):
    model = hydra.utils.instantiate(conf.module)
    dm = hydra.utils.instantiate(conf.data_module)
    print("======== MODEL ========")
    print(model)
    print("======== DATASET ========")
    total_size = 0
    total_events = 0
    for split in ["train", "val", "test"]:
        if not hasattr(dm, f"{split}_data"):
            continue
        dataset = getattr(dm, f"{split}_data")
        ts_field = dataset.timestamps_field
        labels_field = dataset.labels_field
        print(f"SPLIT {split}")
        print(f"  Size: {len(dataset)}")

        lengths = []
        labels = set()
        length_metric = Metric()
        time_delta_metric = Metric()
        duration_metric = Metric()
        for v in dataset:
            l = len(v[ts_field])
            lengths.append(l)
            length_metric.update(l)
            time_delta_metric.update(v[ts_field][1:] - v[ts_field][:-1])
            duration_metric.update(v[ts_field][-1] - v[ts_field][0])
            labels.update(v[labels_field].tolist())

        total_size += len(dataset)
        total_events += sum(lengths)
        print(f"  Num Events: {sum(lengths)}")
        print(f"  Num Labels: {len(labels)}")
        print(f"  Max label: {max(labels)}")

        metrics = length_metric.compute()
        print(f"  Min seq. length: {metrics['min']}")
        print(f"  Max seq. length: {metrics['max']}")
        print(f"  Avg seq. length: {metrics['avg']}")
        print(f"  Median seq. length: {metrics['median']}")

        metrics = duration_metric.compute()
        print(f"  Min duration: {metrics['min']}")
        print(f"  Max duration: {metrics['max']}")
        print(f"  Avg duration: {metrics['avg']}")
        print(f"  Median duration: {metrics['median']}")

        metrics = time_delta_metric.compute()
        print(f"  Min time delta: {metrics['min']}")
        print(f"  Max time delta: {metrics['max']}")
        print(f"  Avg time delta: {metrics['avg']}")
        print(f"  Median time delta: {metrics['median']}")

        try:
            horizon = getattr(model, f"_{split}_metric").horizon
            hor_lens = get_horizon_lengths(dataset, horizon)
            print(f"  Min horizon length: {hor_lens.min().item()}")
            print(f"  Max horizon length: {hor_lens.max().item()}")
            print(f"  Avg horizon length: {hor_lens.float().mean().item()}")
            print(f"  Median horizon length: {hor_lens.float().median().item()}")
        except AttributeError:
            pass
    print(f"TOTAL Size: {total_size}")
    print(f"TOTAL Events: {total_events}")


if __name__ == "__main__":
    main()
