#!/bin/bash

for i in {1..5}
do
    python main.py $i &> log/cv${i}.log
done
