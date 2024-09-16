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

gene_sort_crit = sys.argv[1]
n_bs = sys.argv[2]
split_id = sys.argv[3]
if gene_sort_crit != '20p' and gene_sort_crit != '1000':
    print ("Invalid gene sort criteria")
    exit (1)
if int(n_bs) > 500:
    print ("n_bs > 500 not possible")
    exit (1)


all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/lasso_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + ".tsv", sep='\s+').set_index("SUBJID")
# print (all_tissue_res)
all_tissue_nat_d = all_tissue_res.query("DTHHRDY == 0")
exclude_cols = ['AGE', 'SEX', 'DTHHRDY']
subset_cols = [col for col in all_tissue_res.columns if col not in exclude_cols]

# scaler = StandardScaler()
# all_tissue_res[subset_cols] = scaler.fit_transform(all_tissue_res[subset_cols])

data_points = all_tissue_res[subset_cols].values.flatten()
data_points = data_points[~np.isnan(data_points)]
age_values = all_tissue_res['AGE'].repeat(all_tissue_res[subset_cols].count(axis=1)).values

# Fit a normal distribution to the data
# mu, std = norm.fit(data_points)

df_agegap = pd.DataFrame()
df_agegap["agegap"] = data_points
df_agegap["age"] = age_values

plt.xlim (20, 100)
plt.title("Age vs AgeGap for natural deaths")
sns.scatterplot(data=df_agegap, x="age", y="agegap", 
                hue="agegap", palette='coolwarm', hue_norm=(-12,12), 
                # style="DTHHRDY",  
                # markers={0: "o", 1: "X", 2: "o", 3: "o", 4: "*"}
                )
plt.show()
# plt.savefig('gtex/logistic_PTyj_noGS_C10_tstScale_train_bs10.png')
# plt.savefig("gtex_outputs/lasso_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + tissue + ".png")
plt.clf()