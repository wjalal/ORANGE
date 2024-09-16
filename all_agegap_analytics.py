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
split_id = sys.argv[3]
regr = sys.argv[4]

if gene_sort_crit != '20p' and gene_sort_crit != '1000':
    print ("Invalid gene sort criteria")
    exit (1)
if int(n_bs) > 500:
    print ("n_bs > 500 not possible")
    exit (1)

if regr == "lasso":
    all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/lasso_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + ".tsv", sep='\s+').set_index("SUBJID")
elif regr == "logistic":
    all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/logistic_PTyj_f1ma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + ".tsv", sep='\s+').set_index("SUBJID")

# print (all_tissue_res)
exclude_cols = ['AGE', 'SEX', 'DTHHRDY']

subset_cols = [col for col in all_tissue_res.columns if col not in exclude_cols]

# scaler = StandardScaler()
# scaler = MinMaxScaler(feature_range=(-7,7))
# all_tissue_res[subset_cols] = scaler.fit_transform(all_tissue_res[subset_cols])

all_tissue_res['max_agegap'] = all_tissue_res[subset_cols].max(axis=1)
all_tissue_res['min_agegap'] = all_tissue_res[subset_cols].min(axis=1)

result = agegap_dist_analytics (all_tissue_res, ['max_agegap', 'min_agegap'], gene_sort_crit, n_bs, split_id, regr, True)

print (result)