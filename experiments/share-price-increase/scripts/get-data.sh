#!/usr/bin/env bash

mkdir -p data/src

curl -L -o data/src/SharePriceIncrease.zip https://www.timeseriesclassification.com/aeon-toolkit/SharePriceIncrease.zip
unzip -j data/src/SharePriceIncrease.zip 'SharePriceIncrease_*.arff' -d data
