#!/bin/bash
c="20p"
b="20"
clsp="cl1sp"
train=false
test=true

regr=("pls")

for i in {206201..206225}; do
    if [ "$train" = true ]; then
        python3 stratified_split_dthhrdy.py "$c" "$i"
    fi
    # if [ "$train" = true ]; then
    #     python3 common_test.py "$c" "$i"
    # fi
    for r in "${regr[@]}"; do
        if [ "$train" = true ]; then
            python3 leave-p-out-train-test.py "$c" "$b" "$i" "$r" train
            python3 coeff_gtex_train.py "$c" "$b" "${clsp}${i}" "$r" lpo remove
        else
            python3 leave-p-out-train-test.py "$c" "$b" "$i" "$r"
        fi
        rm -r "gtex/train_splits/train_bs${b}_${clsp}${i}_"*
        python3 all_agegap_analytics.py "$c" "$b" "${clsp}${i}" "$r" > "gtex_outputs/clsp_analytics_rec_${r}_${c}_${i}.txt"
    done
done

wait
