# Preparation
1. Prepare the datafiles via EventStream-GPT:
```
https://github.com/mmcdermott/EventStreamGPT/tree/main
```
2. Create `data` folder and place `events_df.parquet` and `dynamic_measurements_df.parquet` to this folder.
3. Install ICD codes mapping library:
```
pip install git+https://github.com/xiyori/ICD-Mappings.git
```
4. Run data preparation script:
```
spark-submit --master 'local[8]' scripts/make-dataset.py
```
