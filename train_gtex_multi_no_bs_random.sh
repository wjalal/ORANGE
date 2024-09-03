#!/bin/bash
i=$1
# cp train_gtex_no_bs.py train_gtex_multi_no_bs.py
cp test_gtex_train.py test_gtex_train_multi_no_bs.py
cp gtex/GTExTissueAge.py "gtex/GTExTissueAge$i.py"
# cd ../../../gtex/proc
# ./gtex_to_organage_coronary_artery.sh "$i"
# cd ../../organ_aging_proteomics/OrganAge_test/organage
# sed -i "s/.TRAIN.tsv/.TRAIN.$i.tsv/" train_gtex_multi_no_bs.py
# sed -i "s/train_no_bs/train_no_bs_$i/" train_gtex_multi_no_bs.py
sed -i "s/artery_coronary.TEST.tsv/artery_coronary.TEST.$i.tsv/" test_gtex_train_multi_no_bs.py
sed -i "s/GTExTissueAgeBootstrap/GTExTissueAge$i/" test_gtex_train_multi_no_bs.py
# mkdir -p gtex/train_no_bs_$i/data/ml_models/gtexV8/HC/Zprot_perf95/artery_coronary
# touch gtex/train_no_bs_$i/__init__.py
# touch gtex/train_no_bs_$i/data/__init__.py
# touch gtex/train_no_bs_$i/data/ml_models/__init__.py
# touch gtex/train_no_bs_$i/data/ml_models/gtexV8/__init__.py
# touch gtex/train_no_bs_$i/data/ml_models/gtexV8/HC/__init__.py
# touch gtex/train_no_bs_$i/data/ml_models/gtexV8/HC/Zprot_perf95/__init__.py
# touch gtex/train_no_bs_$i/data/ml_models/gtexV8/HC/Zprot_perf95/artery_coronary/__init__.py
# python3 train_gtex_multi_no_bs.py
sed -i "s/train_no_bs/train_no_bs_$i/" "gtex/GTExTissueAge$i.py"
sed -i "s/train_bs10/train_no_bs_$i/" test_gtex_train_multi_no_bs.py
python3 test_gtex_train_multi_no_bs.py