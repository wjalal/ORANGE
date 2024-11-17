#!/bin/bash
c="20p"
b="20"
clsp="cmn"
train=false
test=true

regr=("pls")

for i in {204201..204225}; do
    # if [ "$train" = true ]; then
    #     python3 stratified_split_dthhrdy.py "$c" "$i"
    # fi
    if [ "$train" = true ]; then
        python3 common_test.py "$c" "$i"
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
