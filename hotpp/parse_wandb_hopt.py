import argparse
import wandb
import yaml
import numpy as np
from collections import defaultdict
from numbers import Number

parser = argparse.ArgumentParser("""Get top performing hyper-parameters for a wandb sweep""")
parser.add_argument("sweep_config")
parser.add_argument("sweep_id")
parser.add_argument("-t", "--top-list", help="Comma separated list of top ranges to parse",
                    default="1,3,5")
parser.add_argument("--as-table", action="store_true")
args = parser.parse_args()
top_ks = list(sorted(map(int, args.top_list.split(",")), reverse=True))
top_k = max(top_ks)

with open(args.sweep_config) as fp:
    config = yaml.safe_load(fp)

project = config["project"]
order = "summary_metrics." + config["metric"]["name"]
if config["metric"]["goal"] == "maximize":
    order = "-" + order

api = wandb.Api()
sweep_id = args.sweep_id
if "/" in sweep_id:
    sweep_id = sweep_id.split("/")[-1]
runs = api.runs(path=project,
                filters={"sweep": sweep_id},
                order=order,
                per_page=top_k)
by_metric = defaultdict(list)
for i in range(top_k):
    run = runs[i]
    for metric in config["parameters"]:
        by_metric[metric].append(run.config[metric])

def top_popular(values):
    counts = defaultdict(int)
    for v in values:
        counts[v] += 1
    return max(counts.items(), key=lambda pair: pair[1])[0]

if args.as_table:
    header = ["Parameter"] + [f"Top-{k} mean" for k in top_ks]
    print("\t".join(header))
for metric, values in by_metric.items():
    aggregates = []
    for k in top_ks:
        if isinstance(values[0], Number):
            value = np.mean(values[:k])
            if args.as_table:
                value = str(value)
        else:
            value = top_popular(values[:k])
        aggregates.append(value)
    if args.as_table:
        print("\t".join([metric] + aggregates))
    else:
        print(metric)
        for k, v in zip(top_ks, aggregates):
            print(f"  top-{k}: {v}")
