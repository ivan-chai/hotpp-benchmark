#!/bin/bash

service postgresql start
createdb mimiciv

if [ "$1" == "raw" ]; then
    mode=""
else
    mode="_$1"
fi

cd mimic-code
psql -d mimiciv -f mimic-iv/buildmimic/postgres/create.sql
psql -d mimiciv -v ON_ERROR_STOP=1 -v mimic_data_dir=/data/mimiciv/2.2 -f "mimic-iv/buildmimic/postgres/load${mode}.sql"
psql -d mimiciv -v ON_ERROR_STOP=1 -v mimic_data_dir=/data.mimiciv/2.2 -f mimic-iv/buildmimic/postgres/constraint.sql
psql -d mimiciv -v ON_ERROR_STOP=1 -v mimic_data_dir=/data/mimiciv/2.2 -f mimic-iv/buildmimic/postgres/index.sql
