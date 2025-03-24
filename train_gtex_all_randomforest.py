import pandas as pd
import numpy as np
import warnings
np.warnings = warnings
from sklearn import preprocessing
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor
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
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV

def Train_all_tissue_aging_model_randomforest(md_hot_train, df_prot_train,
                                 seed_list, 
                                 performance_CUTOFF, train_cohort,
                                 norm, agerange, n_bs, split_id,  NPOOL=15):
    # final lists for output
    all_coef_dfs = []   
    NUM_BOOTSTRAP = int(n_bs)
    seed_list = seed_list['BS_Seed']
    seed_list = seed_list[:NUM_BOOTSTRAP]
    print(seed_list)

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
        scaler = PowerTransformer(method='yeo-johnson')
        scaler.fit(df_prot_train_tissue)
        tmp = scaler.transform(df_prot_train_tissue)
        df_prot_train_tissue = pd.DataFrame(tmp, index=df_prot_train_tissue.index, columns=df_prot_train_tissue.columns)
        print (df_prot_train_tissue)

        # save the scaler
        path = 'gtex/train_splits/train_bs1_' + split_id + '/data/ml_models/'+train_cohort+'/'+agerange+'/'+norm+'/'+tissue
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

        print ("starting bootstrap training...")
        pool = mp.Pool(NPOOL)
        input_list = [([df_X_train, df_Y_train, train_cohort,
                        tissue, performance_CUTOFF, norm, agerange,  n_bs, split_id, seed_list[0]])]        
        coef_list = pool.starmap(Train, input_list)
        pool.close()
        pool.join()
        
    dfcoef=[]
    return dfcoef
  
    
def Train(df_X_train, df_Y_train, train_cohort,
              tissue, performance_CUTOFF, norm, agerange, n_bs, split_id, seed):
    
    #setup
    # LASSO
    print (f"starting randomforest?...")
    
    alphas = np.logspace(-3, 1, 100)
    tuned_parameters = [{'alpha': alphas}]
    n_folds=4
    print("initialised randomforest params setup..")

    rf = RandomForestRegressor(random_state=69420, n_estimators=20, max_features=1.0)


    #     # Defining range of components to try in grid search
    # estimators = np.rint(np.linspace (5, 50, 10)).astype(int)  # limiting to a max of 20 components or fewer
    # print (estimators) 
    # tuned_parameters = [{'n_estimators': estimators}]
    # n_folds = 4
    # print("initialised rf params setup... (seed = ", seed, ")")
    
    # # Grid Search
    # clf = GridSearchCV(rf, tuned_parameters, cv=n_folds, scoring="neg_mean_absolute_error", refit=False)
    
    # print("gridSearch done... (seed = ", seed, ")")
    # clf.fit(df_X_train.to_numpy(), df_Y_train.to_numpy().ravel())
    # print("gridSearch fitting done... (seed = ", seed, ")")
    
    # gsdf = pd.DataFrame(clf.cv_results_)
    
    # print("Plot and Pick STARTING :(... (seed = ", seed, ")")
    # best_n_estimators = Plot_and_pick_n_estimators(gsdf, performance_CUTOFF, plot=False)  # Pick best component count
    # print("Plot and Pick done... (seed = ", seed, ")")
    
    # # Retrain with best parameters
    # rf = RandomForestRegressor(random_state=69420, n_estimators=best_n_estimators)


    rf.fit(df_X_train.to_numpy(), df_Y_train.to_numpy().ravel())
    print ("lasso retrained...")
    # SAVE MODEL
    savefp="gtex/train_splits/train_bs1_" + split_id + "/data/ml_models/"+train_cohort+"/"+agerange+"/"+norm+"/"+tissue+"/"+train_cohort+"_"+agerange+"_"+norm+"_randomforest_"+tissue+"_seed"+str(seed)+"_aging_model.pkl"
    pickle.dump(rf, open(savefp, 'wb'))
    # SAVE coefficients            
    coef_list = []

    return coef_list
    


def Plot_and_pick_n_estimators(gsdf, performance_CUTOFF, plot=True):
    """
    This function selects the best number of components based on the performance cutoff and plots the results.
    
    Parameters:
    - gsdf: DataFrame containing the GridSearchCV results
    - performance_CUTOFF: The threshold to determine the acceptable performance range
    - plot: Whether to plot the results (default is True)
    
    Returns:
    - best_n_estimators: The best number of components based on the performance criteria
    """
    
    # Normalize the mean test score
    gsdf["mean_test_score_norm"] = NormalizeData(gsdf["mean_test_score"])
    print("P&P normalized...")

    # Calculate the difference between normalized scores and the performance cutoff
    gsdf["mean_test_score_norm_minus_cutoff"] = gsdf["mean_test_score_norm"] - performance_CUTOFF
    print("P&P cutoff applied...")

    # Get the absolute difference for comparison
    gsdf["mean_test_score_norm_minus_cutoff_abs"] = np.abs(gsdf["mean_test_score_norm_minus_cutoff"])
    print("P&P calculating absolute differences...")

    # Calculate the derivative of performance by number of components
    x = gsdf.param_n_estimators.to_numpy()
    y = gsdf.mean_test_score_norm.to_numpy()
    dx = np.diff(x).mean()  # Step size based on the range of n_estimators
    gsdf["derivative"] = np.gradient(y, dx)
    print("P&P derivative calculated...")

    # Find the number of components with a negative derivative and closest to the performance cutoff
    tmp = gsdf.loc[gsdf.derivative < 0]
    if len(tmp) != 0:
        best_n_estimators = list(tmp.loc[tmp.mean_test_score_norm_minus_cutoff_abs == np.min(tmp.mean_test_score_norm_minus_cutoff_abs)].param_n_estimators)[-1]
    else:
        print('No component count with negative derivative, selecting closest to cutoff.')
        tmp2 = gsdf
        best_n_estimators = list(tmp2.loc[tmp2.mean_test_score_norm_minus_cutoff_abs == np.min(tmp2.mean_test_score_norm_minus_cutoff_abs)].param_n_estimators)[0]
    
    # Plot the results
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(7, 3))

        # Plot normalized test score vs number of components
        sns.scatterplot(data=gsdf, x="param_n_estimators", y="mean_test_score_norm", ax=axs[0])
        sns.scatterplot(data=gsdf.loc[gsdf.param_n_estimators == best_n_estimators], x="param_n_estimators", y="mean_test_score_norm", ax=axs[0])

        sns.scatterplot(data=gsdf, x="param_n_estimators", y="mean_test_score_norm", ax=axs[1])
        sns.scatterplot(data=gsdf.loc[gsdf.param_n_estimators == best_n_estimators], x="param_n_estimators", y="mean_test_score_norm", ax=axs[1])

        axs[0].set_xlim(-0.02, best_n_estimators + 0.1)
        axs[0].set_ylim(0.8, 1.05)
        axs[0].axvline(best_n_estimators, color="red", linestyle="--")
        axs[0].axhline(performance_CUTOFF, color="blue", linestyle="--")

        plt.tight_layout()
        plt.show()

    return best_n_estimators

    
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
        
    def df_prot_train (tissue):
        return pd.read_csv(filepath_or_buffer="proc/proc_datav10/reduced/corr" + gene_sort_crit + "/"+tissue+".TRAIN." + split_id + ".tsv", sep='\s+').set_index("Name")

    from md_age_ordering import return_md_hot
    md_hot_train = return_md_hot()

    bs_seed_list = json.load(open("gtex/Bootstrap_and_permutation_500_seed_dict_500.json"))

    #95% performance
    start_time = time.time()
    dfcoef = Train_all_tissue_aging_model_randomforest(md_hot_train, #meta data dataframe with age and sex (binary) as columns
                                        df_prot_train, #protein expression dataframe returning method (by tissue)
                                        bs_seed_list, #bootstrap seeds
                                        performance_CUTOFF=performance_CUTOFF, #heuristic for model simplification
                                        NPOOL=1, #parallelize
                                        
                                        train_cohort=train_cohort, #these three variables for file naming
                                        norm=norm, 
                                        agerange=agerange, 
                                        n_bs=n_bs,
                                        split_id=split_id
                                        )
    print((time.time() - start_time)/60)
