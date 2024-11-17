import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
import numpy as np
import sys 
from importlib import resources
import warnings
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import norm
from matplotlib.colors import ListedColormap, BoundaryNorm
from CondProbDthHrdy import *
from agegap_analytics import *

gene_sort_crit = sys.argv[1]
n_bs = sys.argv[2]
sp_st = sys.argv[3]
split_id_r1 = sys.argv[4]
split_id_r2 = sys.argv[5]
regr = sys.argv[6]
if len(sys.argv) >= 8:
    agg = sys.argv[7]  #lpo
else:
    agg=""

if gene_sort_crit != '20p' and gene_sort_crit != '1000' and gene_sort_crit != 'deg' and gene_sort_crit != 'AA':
    print ("Invalid gene sort criteria")
    exit (1)
if int(n_bs) > 500:
    print ("n_bs > 500 not possible")
    exit (1)

tissues = [
    "liver",
    "artery_aorta",
    "artery_coronary",
    "brain_cortex",
    "brain_cerebellum",
    # "adrenal_gland",
    "heart_atrial_appendage",
    # "pituitary",
    "adipose_subcutaneous",
    "lung",
    "skin_sun_exposed_lower_leg",
    "nerve_tibial",
    "colon_sigmoid",
    "pancreas",
#    "breast_mammary_tissue",
#    "prostate",
]

with open('gtex/organ_list.dat', 'r') as file:
    tissue_agegaps = ["agegap_" + line.strip() for line in file]

# cols = ['max_agegap', 'min_agegap']
cols = ['max_agegap'] + tissue_agegaps

all_tissue_dth_agegap = {}
for col in cols:
    all_tissue_dth_agegap[col] = []
    for k in range (0,5):
        all_tissue_dth_agegap[col].append({
            'p_gt' : 0,
            'p_lt' : 0,
            'p_r' : 0,
            'p_d' : 0,
        })
        
for s in range (int(split_id_r1), int(split_id_r2)+1):
    split_id = sp_st + str(s)
    if regr == "lasso":
        all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/lasso_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + agg + ".tsv", sep='\t').set_index("SUBJID")
    elif regr == "ridge":
        all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/ridge_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id +  agg + ".tsv", sep='\t').set_index("SUBJID")
    elif regr == "elasticnet":
        all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/elasticnet_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + agg + ".tsv", sep='\t').set_index("SUBJID")
    elif regr == "logistic":
        all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/logistic_PTyj_f1ma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + agg + ".tsv", sep='\t').set_index("SUBJID")
    elif regr == "randomforest":
        all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/randomforest_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + agg + ".tsv", sep='\t').set_index("SUBJID")
    elif regr == "svr":
        all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/svr_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + agg + ".tsv", sep='\t').set_index("SUBJID")
    elif regr == "pls":
        all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/pls_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + agg + ".tsv", sep='\t').set_index("SUBJID")# print (all_tissue_res)
    exclude_cols = ['AGE', 'SEX', 'DTHHRDY']

    subset_cols = [col for col in all_tissue_res.columns if col not in exclude_cols and col[7:] in tissues]
    print (subset_cols)
    all_tissue_res = all_tissue_res.drop(columns=[col for col in  all_tissue_res.columns if col[7:] not in tissues and col not in exclude_cols])

        # Set the minimum number of non-null columns required per row
    # min_non_null_columns = int(len(tissues)/2)
    min_non_null_columns = 5
    # Filter rows with at least 6 non-null values
    all_tissue_res = all_tissue_res.dropna(thresh=min_non_null_columns+len(exclude_cols))
    print (all_tissue_res.shape)
    scaler = StandardScaler()
    # scaler = MinMaxScaler(feature_range=(-7,7))
    all_tissue_res[subset_cols] = scaler.fit_transform(all_tissue_res[subset_cols])

    all_tissue_res['max_agegap'] = all_tissue_res[subset_cols].max(axis=1)
    all_tissue_res['min_agegap'] = all_tissue_res[subset_cols].min(axis=1)

    result = agegap_dist_analytics (all_tissue_res, cols, gene_sort_crit, n_bs, split_id, regr, True)
    for col in cols:
        for k in range (0,4):
            all_tissue_dth_agegap[col][k]['p_gt'] += result[col][k]['p_gt']
            all_tissue_dth_agegap[col][k]['p_lt'] += result[col][k]['p_lt']
            all_tissue_dth_agegap[col][k]['p_r'] += result[col][k]['p_r']
            all_tissue_dth_agegap[col][k]['p_d'] += result[col][k]['p_d']
            if k == 3:
                all_tissue_dth_agegap[col][k]['p_gt'] += result[col][k+1]['p_gt']
                all_tissue_dth_agegap[col][k]['p_lt'] += result[col][k+1]['p_lt']
                all_tissue_dth_agegap[col][k]['p_r'] += result[col][k+1]['p_r']
                all_tissue_dth_agegap[col][k]['p_d'] += result[col][k+1]['p_d']

for col in all_tissue_dth_agegap:
    print("_______________________________")
    print (col)
    print("_________________")
    for i in range(0,4):
        p_gt = all_tissue_dth_agegap[col][i]['p_gt'] / (int(split_id_r2)-int(split_id_r1)+1)
        p_lt = all_tissue_dth_agegap[col][i]['p_lt'] / (int(split_id_r2)-int(split_id_r1)+1)
        p_d = all_tissue_dth_agegap[col][i]['p_d'] / (int(split_id_r2)-int(split_id_r1)+1)
        p_r = all_tissue_dth_agegap[col][i]['p_r'] / (int(split_id_r2)-int(split_id_r1)+1)

        print (f'avg. p({i}|gt) = {p_gt:.4f}') 
        print (f'avg. p({i}|lt) = {p_lt:.4f}')
        print (f'avg. p({i}|r) = {p_lt:.3f}')  
        print (f'avg. p({i}) = {p_d:.4f}') 
        if round(p_gt,5) == 0 and round(p_lt,5) == 0:
            r = 1
            r_i = 1
        elif p_gt == 0:
            r = 0
            r_i = float('inf')
        elif p_lt == 0:
            r = float('inf')
            r_i = 0
        else:
            r = p_gt/p_lt
            r_i = 1/r

        if r > 1: 
            print (f'Extreme positive agers are {r:.3f} times as likely to have died with dthhrdy={i} than extreme negative agers ')
        elif r < 1:
            print (f'Extreme NEGATIVE agers are {r_i:.3f} times as likely to have died with dthhrdy={i} than extreme positive agers ')
        else:
            print(f'Extreme negative and positive agers are equally likely to have died with dthhrdy={i}')

        if p_gt > p_lt:
            rd = p_gt/p_d
            print (f'Extreme positive agers are {rd:.3f} times as likely to have died with dthhrdy={i} than all others')
        elif p_gt < p_lt:
            rd = p_lt/p_d
            print (f'Extreme NEGATIVE agers are {rd:.3f} times as likely to have died with dthhrdy={i} than all others ')
        else :
            rd = p_lt/p_d
            print (f'Extreme negative AND positive agers both are {rd:.3f} times as likely to have died with dthhrdy={i} than all others ')

        if round(p_gt,5) == 0 and round(p_r,5) == 0:
            rr = 1
            rr_i = 1
        elif p_gt == 0:
            rr = 0
            rr_i = float('inf')
        elif p_r == 0:
            rr = float('inf')
            rr_i = 0
        else:
            rr = p_gt/p_r
            rr_i = 1/rr
        if rr > 1: 
            print (f'Extreme positive agers are {rr:.3f} times as likely to have died with dthhrdy={i} than AVERAGE agers ')
        elif rr < 1:
            print (f'AVERAGE agers are {rr_i:.3f} times as likely to have died with dthhrdy={i} than extreme positive agers ')
        else:
            print(f'Extreme positive agers and AVERAGE agers are equally likely to have died with dthhrdy={i}')

        if round(p_lt,5) == 0 and round(p_r,5) == 0:
            rrl = 1
            rrl_i = 1
        elif p_lt == 0:
            rrl = 0
            rrl_i = float('inf')
        elif p_r == 0:
            rrl = float('inf')
            rrl_i = 0
        else:
            rrl = p_lt/p_r
            rrl_i = 1/rrl
        if rrl > 1: 
            print (f'Extreme NEGATIVE agers are {rrl:.3f} times as likely to have died with dthhrdy={i} than AVERAGE agers ')
        elif rrl < 1:
            print (f'AVERAGE agers are {rrl_i:.3f} times as likely to have died with dthhrdy={i} than extreme NEGATIVE agers ')
        else:
            print(f'Extreme NEGATIVE agers and AVERAGE agers are equally likely to have died with dthhrdy={i}')

        print()
    print("_______________________________")
