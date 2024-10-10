from train_gtex_all_lasso import Train_all_tissue_aging_model_lasso
from train_gtex_all_elasticnet import Train_all_tissue_aging_model_elasticnet
from train_gtex_all_randomforest import Train_all_tissue_aging_model_randomforest
from tissue_agegap_analytics_multi import analyse_tissue_agegaps
import pandas as pd
import numpy as np
import json
import sys
import signal
import os

def to_ternary(n):
    ternary_str = ''
    if n == 0:
        ternary_str = '0'
    while n > 0:
        ternary_str = str(n % 3) + ternary_str
        n //= 3
    return ternary_str.zfill(6)

agerange="HC"
performance_CUTOFF=0.95
norm="Zprot_perf"+str(int(performance_CUTOFF*100))
train_cohort="gtexV8"

gene_sort_crit = sys.argv[1]
n_bs = sys.argv[2]
split_id_r = sys.argv[3]
split_id = "cl1sp" + str(split_id_r)
regr = sys.argv[4]

if gene_sort_crit != '20p' and gene_sort_crit != '1000' and gene_sort_crit != 'deg':
    print ("Invalid gene sort criteria")
    exit (1)
if int(n_bs) > 500:
    print ("n_bs > 500 not possible")
    exit (1)

def df_prot_train (tissue):
    return pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/"+tissue+".TRAIN." + split_id + ".tsv", sep='\s+').set_index("Name")
    # return pd.read_csv(filepath_or_buffer="../../../gtex/gtexv8_coronary_artery_TRAIN.tsv", sep='\s+').set_index("Name")


md_hot_train = pd.read_csv(filepath_or_buffer="../../../gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS-rangemid_int.txt", sep='\s+').set_index("SUBJID")
md_hot_train['AGE'] = md_hot_train['AGE'].astype(float)
bs_seed_list = json.load(open("gtex/Bootstrap_and_permutation_500_seed_dict_500.json"))

file_path = f'gtex_outputs/{regr}_corr{gene_sort_crit}_bs{n_bs}_{split_id}_age_guess.csv'

# Check if file exists
if not os.path.exists(file_path):
    # If the file does not exist, create a new DataFrame
    print(f"File {file_path} does not exist. Creating a new DataFrame.")
    df = pd.DataFrame(columns=['mse', 'r2', 'r2_yhat'])
    df.index.name = "ordering"
else:
    # If the file exists, load the existing DataFrame
    print(f"File {file_path} exists. Loading the DataFrame.")
    df = pd.read_csv(filepath_or_buffer=file_path, sep=',', dtype={'ordering': 'object'}).set_index("ordering")

if len(df['mse']) == 0:
    prev_index = -1
else:
    prev_index = int(df.index[-1], 3)
print (f'Starting from {prev_index+1}')

for t in range (prev_index+1, 729):
    f = to_ternary(t)
    md_hot_train_abcdef = md_hot_train.copy()
    md_hot_train_abcdef.loc[md_hot_train_abcdef['AGE'] == 75, 'AGE'] = 75 + (int(f[5])-1) * 10/3
    md_hot_train_abcdef.loc[md_hot_train_abcdef['AGE'] == 65, 'AGE'] = 65 + (int(f[4])-1) * 10/3
    md_hot_train_abcdef.loc[md_hot_train_abcdef['AGE'] == 55, 'AGE'] = 55 + (int(f[3])-1) * 10/3
    md_hot_train_abcdef.loc[md_hot_train_abcdef['AGE'] == 45, 'AGE'] = 45 + (int(f[2])-1) * 10/3
    md_hot_train_abcdef.loc[md_hot_train_abcdef['AGE'] == 35, 'AGE'] = 35 + (int(f[1])-1) * 10/3
    md_hot_train_abcdef.loc[md_hot_train_abcdef['AGE'] == 25, 'AGE'] = 25 + (int(f[0])-1) * 10/3
    # print(md_hot_train_abcdef.head())
    if regr == "lasso":
        Train_all_tissue_aging_model_lasso(md_hot_train_abcdef, #meta data dataframe with age and sex (binary) as columns
                        df_prot_train, #protein expression dataframe returning method (by tissue)
                        bs_seed_list, #bootstrap seeds
                        performance_CUTOFF=performance_CUTOFF, #heuristic for model simplification
                        NPOOL=15, #parallelize
                        train_cohort=train_cohort, #these three variables for file naming
                        norm=norm, 
                        agerange=agerange, 
                        n_bs=n_bs,
                        split_id=split_id
                        )
    elif regr == "elasticnet":
        Train_all_tissue_aging_model_elasticnet(md_hot_train_abcdef, #meta data dataframe with age and sex (binary) as columns
                        df_prot_train, #protein expression dataframe returning method (by tissue)
                        bs_seed_list, #bootstrap seeds
                        performance_CUTOFF=performance_CUTOFF, #heuristic for model simplification
                        NPOOL=15, #parallelize
                        train_cohort=train_cohort, #these three variables for file naming
                        norm=norm, 
                        agerange=agerange, 
                        n_bs=n_bs,
                        split_id=split_id
                        )
    elif regr == "randomforest":
        Train_all_tissue_aging_model_randomforest(md_hot_train_abcdef, #meta data dataframe with age and sex (binary) as columns
                        df_prot_train, #protein expression dataframe returning method (by tissue)
                        bs_seed_list, #bootstrap seeds
                        performance_CUTOFF=performance_CUTOFF, #heuristic for model simplification
                        NPOOL=15, #parallelize
                        train_cohort=train_cohort, #these three variables for file naming
                        norm=norm, 
                        agerange=agerange, 
                        n_bs=n_bs,
                        split_id=split_id
                        )
    metrics = analyse_tissue_agegaps (split_id_r1=split_id_r,
        split_id_r2=split_id_r,
        n_bs=n_bs,
        regr=regr,
        gene_sort_crit=gene_sort_crit,
        curr_ordering=f
    )
    df.loc[f] = list(metrics)
    df.to_csv(f'gtex_outputs/{regr}_corr{gene_sort_crit}_bs{n_bs}_{split_id}_age_guess.csv', index=True)  # Set index=True to include the index in the file

signal.pause()  
