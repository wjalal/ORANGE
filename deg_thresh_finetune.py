import sys
import pandas as pd
import numpy as np
import deg
import pick_deg
from train_gtex_all_pls import Train_tissue_aging_model_pls
import test_gtex_train
import common_test
import json
     
# gene_sort_crit = sys.argv[1]
split_id = sys.argv[1]
n_bs = sys.argv[2]
regr = sys.argv[3]
if len(sys.argv) >= 5:
    lpo_sp = sys.argv[4]
else:
    lpo_sp = ""

metrics = pd.read_csv(filepath_or_buffer=f"gtex_outputs/{regr}_metrics_redcdeg_train_bs{n_bs}_cmn{split_id}{lpo_sp}.csv", sep=',')

# Group by 'tissue' and select the row with the lowest MSE in each group
# min_mse_df = metrics.loc[metrics.groupby('tissue')['MSE'].idxmin()]
max_r2_df = metrics.loc[metrics.groupby('tissue')['R_squared'].idxmax()]
max_r2_df.index = max_r2_df['tissue']
max_r2_df = max_r2_df.drop(columns=['tissue', 'split_id'])
# print(min_mse_df)
print(max_r2_df)

fine_tune_tissues = {
    'liver': [111, 130],
    'colon_sigmoid': [91, 110],
    # 'brain_cortex' : [95, 105],
    'artery_coronary' : [71, 90]
}

def df_prot_train (tissue):
    return pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/reduced/corrdeg" + "/"+tissue+".TRAIN.cmn" + split_id + ".tsv", sep='\s+').set_index("Name")
    # return pd.read_csv(filepath_or_buffer="../../../gtex/gtexv8_coronary_artery_TRAIN.tsv", sep='\s+').set_index("Name")

from md_age_ordering import return_md_hot
md_hot_train = return_md_hot()

bs_seed_list = json.load(open("gtex/Bootstrap_and_permutation_500_seed_dict_500.json"))

for s_organ in fine_tune_tissues.keys():
    for threshold in np.arange(fine_tune_tissues[s_organ][0], fine_tune_tissues[s_organ][1], 1):
        # if threshold > 80:
        deg.main(threshold, s_organ, True)
        pick_deg.main(threshold, s_organ, True)
        sys.argv = ['common_test.py', 'deg', split_id]
        common_test.main()
        sys.argv = ['train_gtex_all_pls.py', 'deg', '20', f"cmn{split_id}"]
        Train_tissue_aging_model_pls (
            s_organ, md_hot_train, df_prot_train,
            bs_seed_list, 
            0.95, "gtexV8",
            "Zprot_perf"+str(int(0.95*100)), "HC", n_bs, f"cmn{split_id}", NPOOL=15
        )
        print("Testing on threshold =", threshold)
        sys.argv = ['test_gtex_train.py', 'deg', '20', f"cmn{split_id}", "pls"]
        split_res = test_gtex_train.main(
            main=True, 
            md_hot_organ=None,
            tissue=None,
            delete_model=False,
            deg_th=str(threshold),
	    s_organ=s_organ,
	    s=True
        )
