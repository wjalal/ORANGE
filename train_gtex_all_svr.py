import pandas as pd
import numpy as np
import warnings
np.warnings = warnings
from sklearn import preprocessing
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
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
import sys
# %matplotlib inline
import mkl
mkl.set_num_threads(1)

import multiprocessing as mp
from sklearn.linear_model import Lasso, LogisticRegression, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV


def Train_all_tissue_aging_model_elasticnet(md_hot_train, df_prot_train,
                                 seed_list, 
                                 performance_CUTOFF, train_cohort,
                                 norm, agerange, n_bs, split_id, NPOOL=15):
    NUM_BOOTSTRAP = int(n_bs)
    seed_list = seed_list['BS_Seed']
    seed_list = seed_list[:NUM_BOOTSTRAP]
    print(seed_list)
    # final lists for output
    all_coef_dfs = []   
    
    with open('gtex/organ_list.dat', 'r') as file:
        tissues = [line.strip() for line in file]

    # Subset to tissue proteins, setup dfX/dfY
    for tissue in tissues:
        print ("STARTING TRAINING FOR " + tissue)
        df_prot_train_tissue = df_prot_train (tissue)
        df_prot_train_tissue.index.names = ['SUBJID']
        md_hot_train_tissue = md_hot_train.merge(right = df_prot_train_tissue.index.to_series(), how='inner', left_index=True, right_index=True)

        # zscore
        # scaler = MinMaxScaler(feature_range = (0,1))
        # scaler = RobustScaler()
        # scaler = StandardScaler()
        scaler = PowerTransformer(method='yeo-johnson')
        scaler.fit(df_prot_train_tissue)
        tmp = scaler.transform(df_prot_train_tissue)
        df_prot_train_tissue = pd.DataFrame(tmp, index=df_prot_train_tissue.index, columns=df_prot_train_tissue.columns)
        print (df_prot_train_tissue)

        # save the scaler
        path = 'gtex/train_splits/train_bs' + n_bs + '_' + split_id + '/data/ml_models/'+train_cohort+'/'+agerange+'/'+norm+'/'+tissue
        fn = '/'+train_cohort+'_'+agerange+'_based_'+tissue+'_gene_zscore_scaler.pkl'
        os.makedirs (path, exist_ok=True)
        pickle.dump (scaler, open(path+fn, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        print("z-scaler is ready...")

        # add sex 
        if "SEX" in list(md_hot_train_tissue.columns):
            # print(md_hot_train[["SEX"]])
            df_X_train = pd.concat([md_hot_train_tissue[["SEX"]], df_prot_train_tissue], axis=1)
        else:
            df_X_train = df_prot_train_tissue.copy()
        df_Y_train = md_hot_train_tissue[["AGE"]].copy()
        
        print (df_X_train)

        # Bootstrap training
        print ("starting bootstrap training...")
        pool = mp.Pool(NPOOL)
        input_list = [([df_X_train, df_Y_train, train_cohort,
                        tissue, performance_CUTOFF, norm, agerange, n_bs, split_id] + [seed_list[i]]) for i in range(NUM_BOOTSTRAP)]        
        coef_list = pool.starmap(Bootstrap_train, input_list)
        pool.close()
        pool.join()
        
    dfcoef=[]
    return dfcoef
  
    
from sklearn.svm import SVR

def Bootstrap_train(df_X_train, df_Y_train, train_cohort,
              tissue, performance_CUTOFF, norm, agerange, n_bs, split_id, seed):
    
    #setup
    X_train_sample = df_X_train.sample(frac=1, replace=True, random_state=seed).to_numpy()
    Y_train_sample = df_Y_train.sample(frac=1, replace=True, random_state=seed).to_numpy()    
    print("did bootstrap setup... (seed = ", seed, ")")
    
    # SVR
    print ("starting SVR... (seed = ", seed, ")")
    svr = SVR(kernel='linear', tol=0.01, max_iter=50000)
    Cs = np.logspace(-2, 0, 150)
    tuned_parameters = [{'C': Cs}]
    n_folds = 4
    print("initialised SVR params setup... (seed = ", seed, ")")
    
    clf = GridSearchCV(svr, tuned_parameters, cv=n_folds, scoring="neg_mean_absolute_error", refit=False)

    print("gridSearch done... (seed = ", seed, ")")
    clf.fit(X_train_sample, np.ravel(Y_train_sample))
    print("gridSearch fitting done... (seed = ", seed, ")")
    
    gsdf = pd.DataFrame(clf.cv_results_)    
    print("Plot and Pick STARTING :(... (seed = ", seed, ")")
    best_C = Plot_and_pick_C(gsdf, performance_CUTOFF, plot=False)   # pick best C
    print("Plot and Pick done... (seed = ", seed, ")")
    
    # Retrain 
    svr = SVR(C=best_C, kernel='linear', tol=0.01, max_iter=50000)
    svr.fit(X_train_sample, np.ravel(Y_train_sample))
    print ("SVR retrained.. (seed = ", seed, ")")
    
    # SAVE MODEL
    savefp="gtex/train_splits/train_bs" + n_bs + "_" + split_id + "/data/ml_models/"+train_cohort+"/"+agerange+"/"+norm+"/"+tissue+"/"+train_cohort+"_"+agerange+"_"+norm+"_svr_"+tissue+"_seed"+str(seed)+"_aging_model.pkl"
    pickle.dump(svr, open(savefp, 'wb'))
    
    # SAVE coefficients (SVR doesn't provide coefficients like linear models, but you can save the support vectors)
    coef_list = []
    
    return coef_list

    

def Plot_and_pick_C(gsdf, performance_CUTOFF, plot=True):
    # pick C at 90-95% top performance, negative derivative (higher C)
    gsdf["mean_test_score_norm"] = NormalizeData(gsdf["mean_test_score"])
    print("P&P normalized...")
    gsdf["mean_test_score_norm_minus95"] = gsdf["mean_test_score_norm"] - performance_CUTOFF
    print("P&P normalized and cut off...")
    gsdf["mean_test_score_norm_minus95_abs"] = np.abs(gsdf["mean_test_score_norm_minus95"])
    print("P&P finding derivative...")
    
    # derivative of performance by C
    x = gsdf.param_C.to_numpy()
    y = gsdf.mean_test_score_norm.to_numpy()
    dx = 0.1
    gsdf["derivative"] = np.gradient(y, dx)
    print("P&P FOUND derivative...")
    
    tmp = gsdf.loc[gsdf.derivative < 0]
    if len(tmp) != 0:
        best_C = list(tmp.loc[tmp.mean_test_score_norm_minus95_abs == np.min(tmp.mean_test_score_norm_minus95_abs)].param_C)[0]
    else:
        print('no C with derivative < 0')
        tmp2 = gsdf
        C_list = list(tmp2.loc[tmp2.mean_test_score_norm_minus95_abs == np.min(tmp2.mean_test_score_norm_minus95_abs)].param_C)
        if len(C_list) > 0:
            best_C = C_list[0]
        else:
            print("No valid 'C' found after filtering.")
            # You can either raise an exception here or set best_C to a default value
            best_C = 0.1  # Or handle this case differently
    
    # PLOT
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
    

if __name__ == "__main__":
    agerange="HC"
    performance_CUTOFF=0.95
    norm="Zprot_perf"+str(int(performance_CUTOFF*100))
    train_cohort="gtexv10"

    gene_sort_crit = sys.argv[1]
    n_bs = sys.argv[2]
    split_id = sys.argv[3]
    if gene_sort_crit != '20p' and gene_sort_crit != '1000' and gene_sort_crit != 'deg' and gene_sort_crit != 'oh':
        print ("Invalid gene sort criteria")
        exit (1)
    if int(n_bs) > 500:
        print ("n_bs > 500 not possible")
        exit (1)

    def df_prot_train (tissue):
        return pd.read_csv(filepath_or_buffer="proc/proc_datav10/reduced/corr" + gene_sort_crit + "/"+tissue+".TRAIN." + split_id + ".tsv", sep='\s+').set_index("Name")

    from md_age_ordering import return_md_hot
    md_hot_train = return_md_hot()

    bs_seed_list = json.load(open("gtex/Bootstrap_and_permutation_500_seed_dict_500.json"))

    #95% performance
    start_time = time.time()
    dfcoef = Train_all_tissue_aging_model_elasticnet(md_hot_train, #meta data dataframe with age and sex (binary) as columns
                                        df_prot_train, #protein expression dataframe returning method (by tissue)
                                        bs_seed_list, #bootstrap seeds
                                        performance_CUTOFF=performance_CUTOFF, #heuristic for model simplification
                                        NPOOL=4, #parallelize
                                        
                                        train_cohort=train_cohort, #these three variables for file naming
                                        norm=norm, 
                                        agerange=agerange, 
                                        n_bs=n_bs,
                                        split_id=split_id
                                        )
    print((time.time() - start_time)/60)
