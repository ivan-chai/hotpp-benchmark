## Prepare dataset:
A simple dataset with events grouped in clusters. Most time steps are zero, while some are 1. A simple zero step prediction baseline achieves low predicted MAE error, but also demonstrates low T-mAP quality.

```
spark-submit <spark-options> scripts/make-dataset.py
```

Run training and evalutaion on sequences:
```
python3 -m hotpp.train --config-dir configs --config-name <model>
```
