import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from scipy.stats import zscore
from sklearn import metrics
import json
import pickle
import time
import random 
import os
from adjustText import adjust_text

# %matplotlib inline


import mkl
mkl.set_num_threads(1)

import multiprocessing as mp
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV

def Train_all_tissue_aging_model(md_hot_train, df_prot_train,
                                 seed_list, 
                                 performance_CUTOFF, train_cohort,
                                 norm, agerange, NPOOL=15):
    seed_list = seed_list['BS_Seed']
    NUM_BOOTSTRAP=len(seed_list)
    print(NUM_BOOTSTRAP)
    # final lists for output
    all_coef_dfs = []   
    
    # bootstrap clocks for each tissue
    # for tissue,plist in tissue_plist_dict.items():
    #     if len(plist)>0:            
    #         print(tissue) 
            
    tissue = 'artery_coronary'
    # Subset to tissue proteins, setup dfX/dfY
    df_prot_train_tissue = df_prot_train (tissue)
    df_prot_train_tissue.index.names = ['SUBJID']
    md_hot_train = md_hot_train.merge(right = df_prot_train_tissue.index.to_series(), how='inner', left_index=True, right_index=True)

    # zscore
    # scaler = MinMaxScaler(feature_range = (0,1))
    scaler = RobustScaler()
    scaler.fit(df_prot_train_tissue)
    tmp = scaler.transform(df_prot_train_tissue)
    df_prot_train_tissue = pd.DataFrame(tmp, index=df_prot_train_tissue.index, columns=df_prot_train_tissue.columns)
    print(df_prot_train_tissue)

    # save the scaler
    path = 'gtex/train_bs10/data/ml_models/'+train_cohort+'/'+agerange+'/'+norm+'/'+tissue
    fn = '/'+train_cohort+'_'+agerange+'_based_'+tissue+'_gene_zscore_scaler.pkl'
    # os.makedirs(path)
    pickle.dump(scaler, open(path+fn, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print("gene-tpm z-scaler is ready...")

    # add sex 
    if "SEX" in list(md_hot_train.columns):
        # print(md_hot_train[["SEX"]])
        df_X_train = pd.concat([md_hot_train[["SEX"]], df_prot_train_tissue], axis=1)
    else:
        df_X_train = df_prot_train_tissue.copy()
    df_Y_train = md_hot_train[["AGE"]].copy()
    
    print (df_X_train)

    # Bootstrap training
    print ("starting bootstrap training...")
    pool = mp.Pool(NPOOL)
    input_list = [([df_X_train, df_Y_train, train_cohort,
                    tissue, performance_CUTOFF, norm, agerange] + [seed_list[i]]) for i in range(NUM_BOOTSTRAP)]        
    coef_list = pool.starmap(Bootstrap_train, input_list)
    pool.close()
    pool.join()
    
    # df_tissue_coef = pd.DataFrame(coef_list, columns=["tissue", "BS_Seed", "C", "y_intercept"]+list(df_X_train.columns))
    # all_coef_dfs.append(df_tissue_coef)
    
    dfcoef=[]
    return dfcoef
  
    
def Bootstrap_train(df_X_train, df_Y_train, train_cohort,
              tissue, performance_CUTOFF, norm, agerange, seed):
    
    #setup
    X_train_sample = df_X_train.sample(frac=1, replace=True, random_state=seed).to_numpy()
    Y_train_sample = df_Y_train.sample(frac=1, replace=True, random_state=seed).to_numpy()    
    print("did bootstrap setup... (seed = ", seed, ")")
    
    # LASSO
    print ("starting lasso?... (seed = ", seed, ")")
    logistic = LogisticRegression(penalty='l1', solver='liblinear', C=10, random_state=0,tol=0.01, max_iter=5000)
    # Cs = 1 / np.logspace(-3, 1, 100)
    # tuned_parameters = [{'C': Cs}]
    # n_folds=4
    # print("initialised lasso params setup... (seed = ", seed, ")")
    # clf = GridSearchCV(logistic, tuned_parameters, cv=n_folds, scoring="f1_micro", refit=False)
    # print("gridSearch done... (seed = ", seed, ")")
    # clf.fit(X_train_sample, np.ravel(Y_train_sample))
    # print("gridSearch fitting done... (seed = ", seed, ")")
    # gsdf = pd.DataFrame(clf.cv_results_)    
    # print("Plot and Pick STARTING :(... (seed = ", seed, ")")
    # best_C=Plot_and_pick_C(gsdf, performance_CUTOFF, plot=False)   #pick best alpha
    # print("Plot and Pick done... (seed = ", seed, ")")
    # # Retrain 
    # logistic = LogisticRegression(penalty='l1', solver='liblinear', C=best_C, random_state=0,tol=0.01, max_iter=5000)
    logistic.fit(X_train_sample, np.ravel(Y_train_sample))
    print ("lasso retrained.. (seed = ", seed, ")")
    # SAVE MODEL
    savefp="gtex/train_bs10/data/ml_models/"+train_cohort+"/"+agerange+"/"+norm+"/"+tissue+"/"+train_cohort+"_"+agerange+"_"+norm+"_l1logistic_"+tissue+"_seed"+str(seed)+"_aging_model.pkl"
    # pickle.dump(lasso, open(savefp, 'wb'))
    pickle.dump(logistic, open(savefp, 'wb'))
    # SAVE coefficients            
    # coef_list = [tissue, seed, best_alpha, lasso.intercept_[0]] + list(lasso.coef_)
    # coef_list = [tissue, seed, best_C, logistic.intercept_[0]] + list(logistic.coef_)
    coef_list = []
    return coef_list
    


def Plot_and_pick_C(gsdf, performance_CUTOFF, plot=True):
    
    # Normalize the mean test scores
    gsdf["mean_test_score_norm"] = NormalizeData(gsdf["mean_test_score"])
    print("P&P normalized...")

    # Calculate the difference from the performance cutoff
    gsdf["mean_test_score_norm_minus95"] = gsdf["mean_test_score_norm"] - performance_CUTOFF
    print("P&P normalized and cutoff applied...")

    # Calculate the absolute difference for easy comparison
    gsdf["mean_test_score_norm_minus95_abs"] = np.abs(gsdf["mean_test_score_norm_minus95"])
    print("P&P finding derivative...")

    # Calculate the derivative of the performance with respect to C
    x = gsdf.param_C.to_numpy()
    y = gsdf.mean_test_score_norm.to_numpy()
    dx = 0.1
    gsdf["derivative"] = np.gradient(y, dx)
    print("P&P FOUND derivative...")

    # Pick the best C where derivative is negative and closest to the performance cutoff
    tmp = gsdf.loc[gsdf.derivative < 0]
    if len(tmp) != 0:
        best_C = list(tmp.loc[tmp.mean_test_score_norm_minus95_abs == np.min(tmp.mean_test_score_norm_minus95_abs)].param_C)[-1]
    else:
        print('No C with derivative <0')
        tmp2 = gsdf
        best_C = list(tmp2.loc[tmp2.mean_test_score_norm_minus95_abs == np.min(tmp2.mean_test_score_norm_minus95_abs)].param_C)[-1]

    # Plot if required
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(7, 3))
        sns.scatterplot(data=gsdf, x="param_C", y="mean_test_score_norm", ax=axs[0])
        sns.scatterplot(data=gsdf.loc[gsdf.param_C == best_C], x="param_C", y="mean_test_score_norm", ax=axs[0])
        sns.scatterplot(data=gsdf, x="param_C", y="mean_test_score_norm", ax=axs[1])
        sns.scatterplot(data=gsdf.loc[gsdf.param_C == best_C], x="param_C", y="mean_test_score_norm", ax=axs[1])
        axs[0].set_xlim(-0.02, best_C + 0.1)
        axs[0].set_ylim(0.8, 1.05)
        axs[0].axvline(0.008)
        axs[0].axhline(performance_CUTOFF)
        plt.tight_layout()
        plt.show()

    return best_C
    
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
    


agerange="HC"
performance_CUTOFF=0.95
norm="Zprot_perf"+str(int(performance_CUTOFF*100))
train_cohort="gtexV8"

def df_prot_train (tissue):
    return pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/"+tissue+".TRAIN.tsv", sep='\s+').set_index("Name")
    # return pd.read_csv(filepath_or_buffer="../../../gtex/gtexv8_coronary_artery_TRAIN.tsv", sep='\s+').set_index("Name")


md_hot_train = pd.read_csv(filepath_or_buffer="../../../gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS-rangemid_int.txt", sep='\s+').set_index("SUBJID")
# tissue_plist_dict = json.load(open("train/data/tissue_pproteinlist_5k_dict_gtex_tissue_enriched_fc4_stable_assay_proteins_seqid.json"))
# bs_seed_list = json.load(open("gtex/Bootstrap_and_permutation_500_seed_dict_small.json"))
bs_seed_list = json.load(open("gtex/Bootstrap_and_permutation_500_seed_dict_10.json"))

#95% performance
start_time = time.time()
dfcoef = Train_all_tissue_aging_model(md_hot_train, #meta data dataframe with age and sex (binary) as columns
                                       df_prot_train, #protein expression dataframe returning method (by tissue)
                                       bs_seed_list, #bootstrap seeds
                                       performance_CUTOFF=performance_CUTOFF, #heuristic for model simplification
                                       NPOOL=30, #parallelize
                                       
                                       train_cohort=train_cohort, #these three variables for file naming
                                       norm=norm, 
                                       agerange=agerange, 
                                       )
print((time.time() - start_time)/60)
