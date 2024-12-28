import random
import hydra
import torch


DELTA_Q_VALUES = [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]


def get_horizon_lengths(dataset, horizon):
    lengths = []
    field = dataset.timestamps_field
    for features in dataset:
        ts = features[field]
        indices = torch.arange(len(ts))
        end_indices = torch.searchsorted(ts, ts + horizon, side="left")
        lengths.append((end_indices - indices - 1).clip(min=0))
    return torch.cat(lengths)


class ReservoirSampler:
    def __init__(self, max_size, dtype=None):
        if dtype is None:
            dtype = torch.float32
        self.max_size = max_size
        self.sample = torch.empty(max_size, dtype=dtype)
        self.reset()

    def reset(self):
        self.total = 0
        self.size = 0

    def update(self, values):
        values = values.flatten()
        if self.size < self.max_size:
            n = self.max_size - self.size
            head, values = values[:n], values[n:]
            head = head[torch.randperm(len(head))]
            self.sample[self.size:self.size + len(head)] = head
            self.size += len(head)
            self.total += len(head)
        if len(values) == 0:
            return self
        accept = torch.rand(len(values)) < self.max_size / (self.max_size + 1 + torch.arange(len(values)))
        accepted = values[accept]
        positions = torch.randint(0, self.max_size, [len(accepted)])
        # Scatter is non-deterministic and can result in wrong solution, but for info it is OK.
        self.sample.scatter_(0, positions, accepted)
        self.total += len(values)
        return self

    def get(self):
        return self.sample[:self.size]


class Metric:
    def __init__(self, q_values=None):
        self._min = None
        self._max = None
        self._avgs = []
        self._medians = []
        self._q_values = q_values
        if q_values is not None:
            self._sampler = ReservoirSampler(10000)

    def update(self, x):
        x = torch.as_tensor(x)
        v = x.min().item()
        self._min = min(self._min, v) if self._min is not None else v
        v = x.max().item()
        self._max = max(self._max, v) if self._max is not None else v
        fx = x.float()
        self._avgs.append(fx.mean().item())
        if self._q_values is not None:
            self._sampler.update(fx)

    def compute(self):
        metrics = {
            "min": self._min,
            "max": self._max,
            "avg": torch.tensor(self._avgs).mean().item(),
        }
        if self._q_values is not None:
            q_values = torch.quantile(self._sampler.get(), torch.tensor(self._q_values))
            for q, v in zip(self._q_values, q_values.tolist()):
                metrics[f"q{q}"] = v
        return metrics


@hydra.main(version_base="1.2", config_path=None)
def main(conf):
    random.seed(0)
    torch.manual_seed(0)
    model = hydra.utils.instantiate(conf.module)
    dm = hydra.utils.instantiate(conf.data_module)
    print("======== MODEL ========")
    print(model)
    print("======== PARAMETERS ========")
    for name, layer in model.named_children():
        print(f"  {name}:", sum([p.numel() for p in layer.parameters()]))
    print("Total parameters:", sum([p.numel() for p in model.parameters()]))
    print("======== DATASET ========")
    total_size = 0
    total_events = 0
    for split in ["train", "val", "test"]:
        if not hasattr(dm, f"{split}_data"):
            continue
        dataset = getattr(dm, f"{split}_data")
        ts_field = dataset.timestamps_field
        print(f"SPLIT {split}")
        print(f"  Size: {len(dataset)}")

        lengths = []
        min_time = 1e9
        max_time = -1e9
        length_metric = Metric(q_values=[0.5])
        time_delta_metric = Metric(q_values=DELTA_Q_VALUES)
        duration_metric = Metric(q_values=[0.5])
        for v in dataset:
            l = len(v[ts_field])
            lengths.append(l)
            length_metric.update(l)
            if l > 1:
                time_delta_metric.update(v[ts_field][1:] - v[ts_field][:-1])
                duration_metric.update(v[ts_field][-1] - v[ts_field][0])
            min_time = min(min_time, v[ts_field].min().item())
            max_time = max(max_time, v[ts_field].max().item())

        total_size += len(dataset)
        total_events += sum(lengths)
        print(f"  Num Events: {sum(lengths)}")
        print(f"  Min timestamp: {min_time}")
        print(f"  Max timestamp: {max_time}")

        metrics = length_metric.compute()
        print(f"  Sequence length")
        print(f"    Min: {metrics['min']}")
        print(f"    Max: {metrics['max']}")
        print(f"    Avg: {metrics['avg']}")
        print(f"    Median: {metrics['q0.5']}")

        metrics = duration_metric.compute()
        print(f"  Duration")
        print(f"    Min: {metrics['min']}")
        print(f"    Max: {metrics['max']}")
        print(f"    Avg: {metrics['avg']}")
        print(f"    Median: {metrics['q0.5']}")

        metrics = time_delta_metric.compute()
        print(f"  Time delta")
        print(f"    Min: {metrics['min']}")
        print(f"    Max: {metrics['max']}")
        print(f"    Avg: {metrics['avg']}")
        for q in DELTA_Q_VALUES:
            v = metrics[f'q{q}']
            print(f"    Q{q}: {v}")

        try:
            horizon = getattr(model, f"_{split}_metric").horizon
            hor_lens = get_horizon_lengths(dataset, horizon)
            print(f"  Horizon length")
            print(f"    Min: {hor_lens.min().item()}")
            print(f"    Max: {hor_lens.max().item()}")
            print(f"    Avg: {hor_lens.float().mean().item()}")
            print(f"    Median: {hor_lens.float().median().item()}")
        except AttributeError:
            pass
    print(f"TOTAL Size: {total_size}")
    print(f"TOTAL Events: {total_events}")


if __name__ == "__main__":
    main()
