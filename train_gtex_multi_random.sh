#!/bin/bash

for i in {7706..7708}; do
    cp train_gtex_all.py train_gtex_multi.py
    cp test_gtex_train.py test_gtex_train_multi.py
    cp gtex/GTExTissueAgeBootstrap.py "gtex/GTExTissueAgeBootstrap$i.py"
    cd ../../../gtex/proc
    ./gtex_to_organage_coronary_artery.sh "$i"
    cd ../../organ_aging_proteomics/OrganAge_test/organage
    sed -i "s/.TRAIN.tsv/.TRAIN.$i.tsv/" train_gtex_multi.py
    sed -i "s/train_bs10/train_bs10_$i/" train_gtex_multi.py
    sed -i "s/artery_coronary.TEST.tsv/artery_coronary.TEST.$i.tsv/" test_gtex_train_multi.py
    sed -i "s/GTExTissueAgeBootstrap/GTExTissueAgeBootstrap$i/" test_gtex_train_multi.py
    mkdir -p gtex/train_bs10_$i/data/ml_models/gtexV8/HC/Zprot_perf95/artery_coronary
    touch gtex/train_bs10_$i/__init__.py
    touch gtex/train_bs10_$i/data/__init__.py
    touch gtex/train_bs10_$i/data/ml_models/__init__.py
    touch gtex/train_bs10_$i/data/ml_models/gtexV8/__init__.py
    touch gtex/train_bs10_$i/data/ml_models/gtexV8/HC/__init__.py
    touch gtex/train_bs10_$i/data/ml_models/gtexV8/HC/Zprot_perf95/__init__.py
    touch gtex/train_bs10_$i/data/ml_models/gtexV8/HC/Zprot_perf95/artery_coronary/__init__.py
    python3 train_gtex_multi.py
    sed -i "s/train_bs10/train_bs10_$i/" "gtex/GTExTissueAgeBootstrap$i.py"
    sed -i "s/train_bs10/train_bs10_$i/" test_gtex_train_multi.py
    python3 test_gtex_train_multi.py
done