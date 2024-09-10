#!/bin/bash

mapfile -t tissues < gtex/organ_list.dat

c="20p"
b="20"
for i in {28501..28501}; do
    cp train_gtex_all_lasso.py train_gtex_multi.py
    cp test_gtex_train.py test_gtex_train_multi.py
    cp gtex/GTExTissueAgeBootstrap.py "gtex/GTExTissueAgeBootstrap$i.py"
    cd ../../../gtex/proc
    for tissue in "${tissues[@]}"; do
        ./gtex_to_organage_corr20p.sh "$i" "$tissue" "$c"
    done
    cd ../../organ_aging_proteomics/OrganAge_test/organage
    sed -i "s/.TRAIN.tsv/.TRAIN.$i.tsv/" train_gtex_multi.py
    sed -i "s/train_bs10/train_bs${b}_$i/" train_gtex_multi.py
    sed -i "s/seed_dict_10/seed_dict_${b}/" train_gtex_multi.py
    sed -i "s/seed_dict_10/seed_dict_${b}/" "gtex/GTExTissueAgeBootstrap$i.py"

    sed -i "s/train_bs10/train_bs${b}_$i/" "gtex/GTExTissueAgeBootstrap$i.py"
    sed -i "s/train_bs10/train_bs${b}_$i/" test_gtex_train_multi.py
    sed -i "s/.TEST.tsv/.TEST.$i.tsv/" test_gtex_train_multi.py
    sed -i "s/GTExTissueAgeBootstrap/GTExTissueAgeBootstrap$i/" test_gtex_train_multi.py
    for tissue in "${tissues[@]}"; do
        mkdir -p "gtex/train_bs${b}_$i/data/ml_models/gtexV8/HC/Zprot_perf95/$tissue"
        touch "gtex/train_bs${b}_$i/__init__.py"
        touch "gtex/train_bs${b}_$i/data/__init__.py"
        touch "gtex/train_bs${b}_$i/data/ml_models/__init__.py"
        touch "gtex/train_bs${b}_$i/data/ml_models/gtexV8/__init__.py"
        touch "gtex/train_bs${b}_$i/data/ml_models/gtexV8/HC/__init__.py"
        touch "gtex/train_bs${b}_$i/data/ml_models/gtexV8/HC/Zprot_perf95/__init__.py"
        touch "gtex/train_bs${b}_$i/data/ml_models/gtexV8/HC/Zprot_perf95/$tissue/__init__.py"
    done
    python3 train_gtex_multi.py $c
    python3 test_gtex_train_multi.py $c
done