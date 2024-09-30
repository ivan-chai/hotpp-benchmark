## Prepare dataset:
A simple dataset with constant timestamp and sequential labels.

In this dataset, ordering is more important than the precise time modeling.

```
spark-submit <spark-options> scripts/make-dataset.py
```

Run training and evalutaion on sequences:
```
python3 -m hotpp.train --config-dir configs --config-name <model>
```
