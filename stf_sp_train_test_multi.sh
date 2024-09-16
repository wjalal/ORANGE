#!/bin/bash
c="20p"
b="20"
clsp="cl1sp"
train=true
regr="lasso"
test=true

for i in {23320..23325}; do
    if [ "$train" = true ]; then
        python3 stratified_split_dthhrdy.py "$c" "$i"
        python3 train_gtex_all_$regr.py "$c" "$b" "${clsp}${i}"
    fi
    if [ "$test" = true ]; then
        python3 test_gtex_train.py "$c" "$b" "${clsp}${i}" "$regr"
    fi
    python3 all_agegap_analytics.py "$c" "$b" "${clsp}${i}" "$regr" > "gtex_outputs/clsp_analytics_rec_${c}_${i}.txt"
done
