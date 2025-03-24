import pandas as pd
import numpy as np
import math
import sys
import test_gtex_train
from sklearn.utils import shuffle
from sklearn.model_selection import LeavePOut
import json 
from train_gtex_all_pls import Train_tissue_aging_model_pls
from train_gtex_all_elasticnet import Train_tissue_aging_model_elasticnet
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, interp1d
from importlib import resources
import pickle
import json
import dill
import warnings
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import norm
from agegap_analytics import *
from sklearn.metrics import mean_squared_error, r2_score
from importlib.resources import files

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)         
np.random.seed(42)

# Define the neural network
class AgeGapNN(nn.Module):
    def __init__(self):
        super(AgeGapNN, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

def train_agegap_nn(df, epochs=300, lr=0.008):
    model = AgeGapNN()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert data to tensors
    age = torch.tensor(df['AGE'].values, dtype=torch.float32).view(-1, 1)
    predicted_age = torch.tensor(df['Predicted_Age'].values, dtype=torch.float32).view(-1, 1)

    # Create a DataLoader
    dataset = TensorDataset(age, predicted_age)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Train the model
    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

    return model

def calculate_lowess_yhat_and_agegap(dfres):
    dfres_agegap = dfres.copy()
    # calculate agegap using lowess of predicted vs chronological age from training cohort
    lowess = sm.nonparametric.lowess
    lowess_fit = lowess(dfres_agegap.Predicted_Age.to_numpy(), dfres_agegap.AGE.to_numpy(), frac=0.8, it=3)
    lowess_fit_int = interp1d(lowess_fit[:,0], lowess_fit[:,1], bounds_error=False, kind='linear', fill_value=(0, 150)) 
    y_lowess = lowess_fit_int(dfres_agegap.AGE)
    dfres_agegap["yhat_lowess"] = y_lowess
    # dfres_agegap["yhat_lowess"] = age_prediction_lowess(np.array(dfres_agegap.Age))
    if len(dfres_agegap.loc[dfres_agegap.yhat_lowess.isna()]) > 0:
        print("Could not predict lowess yhat in " + str(len(dfres_agegap.loc[dfres_agegap.yhat_lowess.isna()])) + " samples")
        dfres_agegap = dfres_agegap.dropna(subset="yhat_lowess")
    dfres_agegap["AgeGap"] = dfres_agegap["Predicted_Age"] - dfres_agegap["yhat_lowess"]
    return dfres_agegap

def calculate_yhat_and_agegap_with_nn(dfres):
    # Train the neural network
    model = train_agegap_nn(dfres)

    # Switch to evaluation mode
    model.eval()
    
    # Generate predictions for yhat_lowess
    age_tensor = torch.tensor(dfres['AGE'].values, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        y_lowess = model(age_tensor).numpy().flatten()

    # Add predictions and calculate AgeGap
    dfres_agegap = dfres.copy()
    dfres_agegap["yhat_lowess"] = y_lowess
    dfres_agegap["AgeGap"] = dfres_agegap["Predicted_Age"] - dfres_agegap["yhat_lowess"]
    
    return dfres_agegap

def leave_p_out_consecutive_by_p(df, P):
    n = df.shape[0]
    splits = []
    r = n % P  # Calculate the remainder

    # Slide the window by P, but handle the last part with P + r
    for i in range(0, n - r, P):
        test_indices = np.arange(i, i + P)
        train_indices = np.concatenate([np.arange(0, i), np.arange(i + P, n)])
        splits.append((train_indices, test_indices))

    # Handle the last chunk with P + r if there's a remainder
    if r != 0:
        test_indices = np.arange(n - P - r, n)
        train_indices = np.arange(0, n - P - r)
        splits.append((train_indices, test_indices))

    return splits

gene_sort_crit = sys.argv[1]
n_bs = sys.argv[2]
rand_seed = sys.argv[3]
regr = sys.argv[4]
train = (len(sys.argv) >= 6 and sys.argv[5] == 'train')

if gene_sort_crit != '20p' and gene_sort_crit != '1000' and gene_sort_crit != 'deg' and gene_sort_crit != 'AA':
    print ("Invalid args")
    exit (1)

with open('gtex/organ_list.dat', 'r') as file:
    organ_list = [line.strip() for line in file]

bs_seed_list = json.load(files("gtex").joinpath("Bootstrap_and_permutation_500_seed_dict_500.json").open("r"))
agerange="HC"
performance_CUTOFF=0.95
norm="Zprot_perf"+str(int(performance_CUTOFF*100))
train_cohort="gtexv10"

from md_age_ordering import return_md_hot
md_hot = return_md_hot()
md_hot['DTHHRDY'] = md_hot['DTHHRDY'].fillna(0)
all_tissue_res = md_hot.copy()

for organ in organ_list:
    print(organ)
    res = pd.DataFrame()

    # Load and prepare the data
    df_gene = pd.read_csv("proc/proc_datav10/reduced/corr" + gene_sort_crit + "/" + organ + ".tsv", sep='\s+').set_index("Name")
    df_gene.index.names = ['SUBJID']
    
    # Merge with metadata
    md_hot_organ = md_hot.merge(right=df_gene.index.to_series(), how='inner', left_index=True, right_index=True)
    df_gene['DTHHRDY'] = md_hot_organ['DTHHRDY']

    # Separate the classes
    df_gene_dthhrdy_1 = df_gene[df_gene['DTHHRDY'] == 1]  # DTHHRDY == 1
    df_gene_remain = df_gene[df_gene['DTHHRDY'] != 1]  # DTHHRDY != 1

    # Step 1: Group by DTHHRDY in df_gene_remain and shuffle within groups
    grouped = df_gene_remain.groupby('DTHHRDY')
    shuffled_groups = [shuffle(group, random_state=int(rand_seed)) for _, group in grouped]
    # print (shuffled_groups)

    # Step 2: Concatenate the shuffled groups and shuffle the final dataframe
    df_gene_remain_homogenized = pd.concat(shuffled_groups).sample(frac=1, random_state=int(rand_seed))

    # Ensure that the index name is preserved
    df_gene_remain_homogenized.index.names = df_gene_remain.index.names

    # # Step 3: Define LeavePOut cross-validator
    # lpo = LeavePOut(p=10)

    # Convert the homogenized dataframe to numpy arrays for the cross-validation loop
    X = df_gene_remain_homogenized.drop(columns=['DTHHRDY'])

    splits = leave_p_out_consecutive_by_p(X, int(X.shape[0]*0.05))
    print(len(splits))

    # Leave-p-out loop
    for i, (train_idx, test_idx) in enumerate(splits):
        # Get train and test splits, preserving the original index
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        # Combine the DTHHRDY == 1 data into the test set
        if i == len(splits) - 1:
            X_test_dthhrdy_1 = df_gene_dthhrdy_1.drop(columns=['DTHHRDY'])
            X_test = pd.concat([X_test, X_test_dthhrdy_1])

        # Ensure the original index names are preserved
        X_train.index.names = df_gene_remain_homogenized.index.names
        X_test.index.names = df_gene_remain_homogenized.index.names

        # Perform your training on X_train, y_train
        # e.g. model.fit(X_train, y_train)

        # Perform your evaluation on X_test, y_test
        # e.g. model.predict(X_test)

        def df_prot_train_f (tissue):
            return X_train

        # print(X_train)
        # print(X_test)
        # pls_predictor = None
        if train:
            md_hot_split = md_hot_organ.merge(right = X_train.index.to_series(), how='inner', left_index=True, right_index=True)
            if regr == "pls":
                Train_tissue_aging_model_pls (
                    tissue=organ,
                    md_hot_train=md_hot_split, #meta data dataframe with age and sex (binary) as columns
                    df_prot_train=df_prot_train_f, #protein expression dataframe returning method (by tissue)
                    seed_list=bs_seed_list, #bootstrap seeds
                    performance_CUTOFF=performance_CUTOFF, #heuristic for model simplification
                    NPOOL=1, #parallelize
                    train_cohort=train_cohort, #these three variables for file naming
                    norm=norm, 
                    agerange=agerange, 
                    n_bs=n_bs,
                    split_id=f"cl1sp{rand_seed}_{i}",
                )
            elif regr == "elasticnet":
                Train_tissue_aging_model_elasticnet (
                    tissue=organ,
                    md_hot_train=md_hot_split, #meta data dataframe with age and sex (binary) as columns
                    df_prot_train=df_prot_train_f, #protein expression dataframe returning method (by tissue)
                    seed_list=bs_seed_list, #bootstrap seeds
                    performance_CUTOFF=performance_CUTOFF, #heuristic for model simplification
                    NPOOL=1, #parallelize
                    train_cohort=train_cohort, #these three variables for file naming
                    norm=norm, 
                    agerange=agerange, 
                    n_bs=n_bs,
                    split_id=f"cl1sp{rand_seed}_{i}",
                )

            sys.argv = ['test_gtex_train.py', gene_sort_crit, n_bs, f"cl1sp{rand_seed}", regr, f"{i}"]
            split_res = test_gtex_train.main(
                main=False, 
                md_hot_organ=md_hot.merge(right=X_test.index.to_series(), how='inner', left_index=True, right_index=True),
                tissue=organ,
                delete_model=False,
            )
            res = pd.concat([res, split_res])

        # Optionally, save your train and test splits for each iteration
        # Save the data for each loop iteration
        # X_train.to_csv("proc/proc_datav10/reduced/corr" + gene_sort_crit + "/" + organ + ".TRAIN.LPO.cl1sp" + rand_seed + ".tsv", sep='\t', index=True)
        # X_test.to_csv("proc/proc_datav10/reduced/corr" + gene_sort_crit + "/" + organ + ".TEST.LPO.cl1sp" + rand_seed + ".tsv", sep='\t', index=True)

        # Break after the first iteration for testing; remove this to run for all splits
    if not train:
        res = pd.read_csv(f"gtex_outputs/{regr}_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + organ + "_" + rand_seed + "lpo" + ".tsv", sep='\t', index_col='SUBJID')

    res = calculate_yhat_and_agegap_with_nn(res)
    # res = calculate_lowess_yhat_and_agegap(res)
    res.to_csv(f"gtex_outputs/{regr}_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + organ + "_" + rand_seed + "lpo" + ".tsv", sep='\t', index=True)
    
    # all_tissue_res.loc[res.index, 'p_age_' + organ] = res["Predicted_Age"]
    all_tissue_res.loc[res.index, 'agegap_' + organ] = res['AgeGap']
    # tissue_res = pd.DataFrame(index=res.index)
    # tissue_res['agegap_' + tissue] = res['AgeGap']
    # tissue_res['DTHHRDY'] = res['DTHHRDY']
    mse = mean_squared_error(res['AGE'], res['Predicted_Age'])
    r2 = r2_score(res['AGE'], res['Predicted_Age'])
    r2_yhat = r2_score(res['AGE'], res['yhat_lowess'])
    print(f'Mean Squared Error: {mse} = ({mse**0.5})^2')
    print(f'R-squared: {r2}')
    print(f'R-squared with y_hat: {r2_yhat}')

    # Prepare the plot data
    toplot = res
    toplot = toplot.sort_values("AGE")

    # Create the plot with very little margin
    plt.figure(figsize=(6, 4.5))
    plt.xlim(20, 80)
    sns.scatterplot(data=toplot, x="AGE", y="Predicted_Age", 
                    hue="AgeGap", palette='coolwarm', hue_norm=(-12, 12), 
                    style="DTHHRDY", markers={0: "o", 1: "X", 2: "o", 3: "o", 4: "o"}, legend=False)

    # Add the quadratic smooth line
    toplot = toplot.drop_duplicates(subset='AGE')
    x_smooth = np.linspace(toplot.AGE.min(), toplot.AGE.max(), 300)
    quadratic_interp = interp1d(toplot.AGE, toplot.yhat_lowess, kind='quadratic')
    y_smooth = quadratic_interp(x_smooth)

    plt.plot(x_smooth, y_smooth, label='Smoothed line', color='black')
    # plt.title(f"Age gap predictions for {organ}")

    
    # Display MSE and R-squared on the plot
    plt.text(0.95, 0.15, f'RMSE={mse**0.5:.2f}\nR²={r2:.2f}', 
            transform=plt.gca().transAxes, ha='right', va='top', fontsize=18,
            bbox=dict(facecolor='white', alpha=0, edgecolor='white', boxstyle='round,pad=0.3'))

    # Adjust layout for very little margin
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    # plt.show()
    # plt.savefig('gtex/logistic_PTyj_noGS_C10_tstScale_train_bs10.png')
    split_id = f"cl1sp{rand_seed}"
    if regr == "lasso":
        plt.savefig("gtex_outputs/lasso_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id +  "_" + organ + ".png")
    elif regr == "ridge":
        plt.savefig("gtex_outputs/ridge_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id +  "_" + organ + ".png")
    elif regr == "elasticnet":
        plt.savefig("gtex_outputs/elasticnet_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + organ + ".png")
    elif regr == "randomforest":
        plt.savefig("gtex_outputs/randomforest_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + organ + ".png")
    elif regr == "l1logistic":
        plt.savefig("gtex_outputs/l1logistic_PTyj_f1ma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + organ + ".png")
    elif regr == "svr":
        plt.savefig("gtex_outputs/svr_PTyj_f1ma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + organ + ".png")
    elif regr == "pls":
        plt.savefig("gtex_outputs/pls_PTyj_f1ma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + organ + ".png")
    plt.clf()
    print() 


exclude_cols = ['AGE', 'SEX', 'DTHHRDY']
subset_cols = [col for col in all_tissue_res.columns if col not in exclude_cols]
all_tissue_res = all_tissue_res.dropna(how='all', subset=subset_cols)

all_tissue_res['non_null_count'] = all_tissue_res.count(axis=1)
all_tissue_res = all_tissue_res.sort_values(by='non_null_count', ascending=False)
all_tissue_res = all_tissue_res.drop(columns=['non_null_count'])


# print (all_tissue_res)
lpo_sp = "lpo"
if regr == "lasso":
    all_tissue_res.to_csv("gtex_outputs/lasso_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + ".tsv", sep='\t', index=True)
elif regr == "ridge":
    all_tissue_res.to_csv("gtex_outputs/ridge_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + ".tsv", sep='\t', index=True)
elif regr == "elasticnet":
    all_tissue_res.to_csv("gtex_outputs/elasticnet_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + ".tsv", sep='\t', index=True)
elif regr == "l1logistic":
    all_tissue_res.to_csv("gtex_outputs/l1logistic_PTyj_f1ma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + ".tsv", sep='\t', index=True)
elif regr == "randomforest":
    all_tissue_res.to_csv("gtex_outputs/randomforest_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + ".tsv", sep='\t', index=True)
elif regr == "svr":
    all_tissue_res.to_csv("gtex_outputs/svr_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + ".tsv", sep='\t', index=True)
elif regr == "pls":
    all_tissue_res.to_csv("gtex_outputs/pls_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + ".tsv", sep='\t', index=True)

print(all_tissue_res)