The dataset contains multiple fields beyond MTPP's "timestamps" and "labels".

TODO: Add more fields.

# Prepare dataset
```bash
spark-submit --driver-memory 16g -c spark.network.timeout=100000s --master local[16] ./scripts/make-dataset.py
```

# Run training
```bash
python3 -m hotpp.train --config-dir configs --config-name detection
```

# Predict embeddings
```bash
python3 -m hotpp.embed --config-dir configs --config-name detection +embeddings_path=embeddings.parquet trainer.devices=1
```

# Downstream evaluation
Under development.
