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
split_id_r1 = sys.argv[3]
split_id_r2 = sys.argv[4]
regr = sys.argv[5]

if gene_sort_crit != '20p' and gene_sort_crit != '1000':
    print ("Invalid gene sort criteria")
    exit (1)
if int(n_bs) > 500:
    print ("n_bs > 500 not possible")
    exit (1)

cols = ['max_agegap', 'min_agegap']
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
    split_id = "cl1sp" + str(s)
    if regr == "lasso":
        all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/lasso_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + ".tsv", sep='\s+').set_index("SUBJID")
    elif regr == "ridge":
        all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/ridge_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + ".tsv", sep='\s+').set_index("SUBJID")
    elif regr == "elasticnet":
        all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/elasticnet_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + ".tsv", sep='\s+').set_index("SUBJID")
    elif regr == "logistic":
        all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/logistic_PTyj_f1ma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + ".tsv", sep='\s+').set_index("SUBJID")
    elif regr == "randomforest":
        all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/randomforest_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + ".tsv", sep='\s+').set_index("SUBJID")
    # print (all_tissue_res)
    exclude_cols = ['AGE', 'SEX', 'DTHHRDY']

    subset_cols = [col for col in all_tissue_res.columns if col not in exclude_cols]

    # scaler = StandardScaler()
    # scaler = MinMaxScaler(feature_range=(-7,7))
    # all_tissue_res[subset_cols] = scaler.fit_transform(all_tissue_res[subset_cols])

    all_tissue_res['max_agegap'] = all_tissue_res[subset_cols].max(axis=1)
    all_tissue_res['min_agegap'] = all_tissue_res[subset_cols].min(axis=1)

    result = agegap_dist_analytics (all_tissue_res, cols, gene_sort_crit, n_bs, split_id, regr, False)
    for col in cols:
        for k in range (0,5):
            all_tissue_dth_agegap[col][k]['p_gt'] += result[col][k]['p_gt']
            all_tissue_dth_agegap[col][k]['p_lt'] += result[col][k]['p_lt']
            all_tissue_dth_agegap[col][k]['p_r'] += result[col][k]['p_r']
            all_tissue_dth_agegap[col][k]['p_d'] += result[col][k]['p_d']

for col in all_tissue_dth_agegap:
    print("_______________________________")
    print (col)
    print("_________________")
    for i in range(0,5):
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
            print (f'Extreme positive agers are {r:.3f} times more likely to have died with dthhrdy={i} than extreme negative agers ')
        elif r < 1:
            print (f'Extreme NEGATIVE agers are {r_i:.3f} times more likely to have died with dthhrdy={i} than extreme positive agers ')
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
            print (f'Extreme positive agers are {rr:.3f} times more likely to have died with dthhrdy={i} than AVERAGE agers ')
        elif rr < 1:
            print (f'AVERAGE agers are {rr_i:.3f} times more likely to have died with dthhrdy={i} than extreme positive agers ')
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
            print (f'Extreme NEGATIVE agers are {rrl:.3f} times more likely to have died with dthhrdy={i} than AVERAGE agers ')
        elif rrl < 1:
            print (f'AVERAGE agers are {rrl_i:.3f} times more likely to have died with dthhrdy={i} than extreme NEGATIVE agers ')
        else:
            print(f'Extreme NEGATIVE agers and AVERAGE agers are equally likely to have died with dthhrdy={i}')

        print()
    print("_______________________________")