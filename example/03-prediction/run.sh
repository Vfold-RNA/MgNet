#!/bin/bash

rm -rf log/*

for i in {1..5}
do
    rm -rf test_result/cv${i}
    python main.py $i ../02-prepare-for-cnn/image/temp_48_small_partial_charge_radius/  &> log/cv${i}.log
done
