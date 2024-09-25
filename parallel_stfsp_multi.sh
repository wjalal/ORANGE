#!/bin/bash
c="20p"
b="1"
clsp="cl1sp"
train=false
regr="randomforest"
test=true

for i in {202101..202115}; do
    (
        if [ "$train" = true ]; then
            python3 stratified_split_dthhrdy.py "$c" "$i"
            python3 train_gtex_all_"$regr".py "$c" "$b" "${clsp}${i}"
        fi
        if [ "$test" = true ]; then
            python3 test_gtex_train.py "$c" "$b" "${clsp}${i}" "$regr"
        fi
        python3 all_agegap_analytics.py "$c" "$b" "${clsp}${i}" "$regr" > "gtex_outputs/clsp_analytics_rec_${c}_${i}.txt"
    ) &
done

wait
