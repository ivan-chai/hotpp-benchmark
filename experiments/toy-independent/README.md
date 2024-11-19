## Prepare dataset:
A simple dataset with independent events.

This dataset requires accurate estimation of prior events distribution on the horizon, rather than dependencies between different event types and event timestamps.

```
spark-submit <spark-options> scripts/make-dataset.py
```

Run training and evalutaion on sequences:
```
python3 -m hotpp.train --config-dir configs --config-name <model>
```
