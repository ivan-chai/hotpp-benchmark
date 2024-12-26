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
python3 -m hotpp.embed --config-dir configs --config-name detection ~logger +embeddings_path=embeddings-last-1000.pkl trainer.devices=1 data_module.batch_size=512 data_module.min_length=1000 data_module.max_length=1000 +reducer=last +data_module.position=last data_module.min_required_length=1'
```
