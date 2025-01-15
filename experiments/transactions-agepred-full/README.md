# Transactions AgePred dataset beyond MTPP
The variant of the dataset in this folder adds "amount" both as a model input and as a model target, along with the simple "timestamps" and "labels" MTPP fields.

## Download and prepare dataset:
```
spark-submit <spark-options> scripts/make-dataset.py
```

Run training and evalutaion on sequences:
```
python3 -m hotpp.train --config-dir configs --config-name <model>
```

Evaluate downstream:
```
python3 -m hotpp.eval_downstream --config-dir configs --config-name next_item [++downstream.num_workers=<workers>]
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

# Predict embeddings
```bash
python3 -m hotpp.embed --config-dir configs --config-name detection +embeddings_path=embeddings.parquet
```
