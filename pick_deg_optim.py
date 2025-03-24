import sys
import pandas as pd
import numpy as np
import deg
import pick_deg
from train_gtex_all_pls import Train_tissue_aging_model_pls
import test_gtex_train
import common_test
import json
     
fine_tune_tissues = {
    'liver': 122,
    'artery_aorta': 90,
    'artery_coronary': 81,
    'brain_cortex': 90,
    'brain_cerebellum': 60,
    # 'adrenal_gland': 0.01,
    'heart_atrial_appendage': 90,
    # 'pituitary': 0.01,
    'adipose_subcutaneous': 120,
    'lung': 160,
    'skin_sun_exposed_lower_leg': 80,
    'nerve_tibial': 100,
    'colon_sigmoid': 100,
    'pancreas': 10,
    # 'breast_mammary_tissue': 0.01,
    # 'prostate': 0.05,
}


from md_age_ordering import return_md_hot
md_hot_train = return_md_hot()

bs_seed_list = json.load(open("gtex/Bootstrap_and_permutation_500_seed_dict_500.json"))

for s_organ in fine_tune_tissues.keys():
    threshold = fine_tune_tissues[s_organ]
    deg.main(threshold, s_organ, True)
    pick_deg.main(threshold, s_organ, True)
