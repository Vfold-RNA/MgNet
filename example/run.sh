#!/bin/bash

cd 02-prepare-for-cnn
bash run.sh &> run.log
cd -

cd 03-prediction
bash run.sh

cd test_result
bash run.sh
