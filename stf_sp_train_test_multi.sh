#!/bin/bash
# c="oh"
c="20p"
b="20"
clsp="cl1sp"
train=true
test=true

# regr=("lasso")
regr=("elasticnet")

for i in {205003..205003}; do

    if [ "$train" = true ]; then
        if [ "$clsp" = "cl1sp" ]; then
            python3 stratified_split_dthhrdy.py "$c" "$i"
        elif [ "$clsp" = "cmn" ]; then
            python3 common_test.py "$c" "$i"
        fi
    fi
    for r in "${regr[@]}"; do
        if [ "$train" = true ]; then
            python3 train_gtex_all_"$r".py "$c" "$b" "${clsp}${i}"
        fi
        if [ "$test" = true ]; then
            python3 test_gtex_train.py "$c" "$b" "${clsp}${i}" "$r"
        fi
        # python3 all_agegap_analytics.py "$c" "$b" "${clsp}${i}" "$r" > "gtex_outputs/clsp_analytics_rec_${r}_${c}_${i}.txt"
    done
done

wait
