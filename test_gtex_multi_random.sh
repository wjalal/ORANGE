#!/bin/bash
c="20p"
b="50"
for i in {23001..23003}; do
    python3 test_gtex_train.py $c $b $i
done