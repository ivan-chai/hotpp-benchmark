A simple time-series classification dataset.

The goal is to predict future prices change.

https://www.timeseriesclassification.com/description.php?Dataset=SharePriceIncrease

# Preprocessing
```bash
sh scripts/get-data.sh

spark-submit ./scripts/make-dataset.py

spark-submit ./scripts/make-dataset.py --chunked
```
