#!/bin/bash

mapfile -t tissues < gtex/organ_list.dat
c="20p"
b="50"
for i in {23001..23010}; do
    cd ../../../gtex/proc
    for tissue in "${tissues[@]}"; do
        ./gtex_to_organage_corr20p.sh "$i" "$tissue" "$c"
    done
    cd ../../organ_aging_proteomics/OrganAge_test/organage

    python3 train_gtex_all_lasso.py $c $b $i
    python3 test_gtex_train.py $c $b $i
done