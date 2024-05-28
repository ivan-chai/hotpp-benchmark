## Download and prepare dataset:
```
spark-submit <spark-options> scripts/make-dataset.py
```

Run training and evalutaion on sequences:
```
python3 -m esp_horizon.train --config-dir configs --config-name <model>
```

Evaluate downstream:
```
python3 -m esp_horizon.eval_downstream --config-dir configs --config-name downstream +model_config=<model> [++num_workers=<workers>]
```

### Useful Spark options
Set memory limit:
```
spark-submit --driver-memory 6g
```

Set the number of threads:
```
spark-submit --master 'local[8]'
```
