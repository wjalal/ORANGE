#!/bin/bash

c="20p"
b="20"
for i in {28501..28501}; do
    cp test_gtex_train.py test_gtex_train_multi.py
    cp gtex/GTExTissueAgeBootstrap.py "gtex/GTExTissueAgeBootstrap$i.py"
    sed -i "s/train_bs10/train_bs${b}_$i/" "gtex/GTExTissueAgeBootstrap$i.py"
    sed -i "s/train_bs10/train_bs${b}_$i/" test_gtex_train_multi.py
    sed -i "s/seed_dict_10/seed_dict_${b}/" "gtex/GTExTissueAgeBootstrap$i.py"
    sed -i "s/.TEST.tsv/.TEST.$i.tsv/" test_gtex_train_multi.py
    sed -i "s/GTExTissueAgeBootstrap/GTExTissueAgeBootstrap$i/" test_gtex_train_multi.py
    python3 test_gtex_train_multi.py $c
done