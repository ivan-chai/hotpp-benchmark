Download and prepare dataset:
```
spark-submit <spark-options> scripts/make-dataset.py <data-root>
```

### Useful spark options
Set memory limit:
```
spark-submit --driver-memory 6g
```

Set the number of threads:
```
spark-submit --master 'local[8]'
```
