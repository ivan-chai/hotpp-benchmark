#!/usr/bin/env bash

mkdir -p data/src

curl -L -o data/src/Earthquakes.zip https://www.timeseriesclassification.com/aeon-toolkit/Earthquakes.zip
unzip -j data/src/Earthquakes.zip 'Earthquakes_*.txt' -d data
