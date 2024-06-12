# Preparation

## 1. Build MIMIC dataset via EventStream-GPT

1. Download MIMIC-IV dataset from the [official site](https://mimic.mit.edu/) and place it in the `hotpp-benchmark/experiments/mimiciv/data` directory.

2. Build a docker image and spin up a container:
```
docker build -t esgpt docker-postgres
docker run -d -v "$(pwd)/data:/data" --name esgpt esgpt tail -f /dev/null
```
3. Run a script to create a postgres database:
```
docker exec esgpt ./createdb.sh <format>
```
where `<format>` specifies the dataset format, either `raw`, `gz`, or `7z`.

Alternatively, access the container interactively and follow the [mimic-code's guide](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres) to create the database manually.

4. Run EventStreamGPT data preprocessing to obtain intermediate parquet files
```
docker exec esgpt psql -c "CREATE USER admin WITH SUPERUSER PASSWORD 'admin';"
docker exec -w /var/lib/postgresql/MIMICIV_FMs_public esgpt ./scripts/build_dataset.sh cohort_name=hotpp_cohort
docker cp esgpt:/var/lib/postgresql/MIMICIV_FMs_public/data/hotpp_cohort/events_df.parquet data/
docker cp esgpt:/var/lib/postgresql/MIMICIV_FMs_public/data/hotpp_cohort/dynamic_measurements_df.parquet data/
```

## 2. Convert to the HOTPP format

1. Install ICD codes mapping library:
```
pip install git+https://github.com/xiyori/ICD-Mappings.git
```
2. Run data preparation script:
```
spark-submit --master 'local[8]' scripts/make-dataset.py
```
