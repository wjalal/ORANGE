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
from sklearn.cross_decomposition import PLSRegression

import multiprocessing as mp
from sklearn.linear_model import Lasso, LogisticRegression, Ridge, ElasticNet, HuberRegressor, QuantileRegressor
from sklearn.model_selection import GridSearchCV

from sdv.single_table import TVAESynthesizer, CTGANSynthesizer, CopulaGANSynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import Metadata
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Train_tissue_aging_model_pls (tissue, md_hot_train, df_prot_train,
                                 seed_list, 
                                 performance_CUTOFF, train_cohort,
                                 norm, agerange, n_bs, split_id, NPOOL=15):
    NUM_BOOTSTRAP = int(n_bs)
    seed_list = seed_list['BS_Seed']
    seed_list = seed_list[:NUM_BOOTSTRAP]
    print(seed_list)
    # final lists for output
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
    # comb_data = pd.concat ([df_X_train, df_Y_train], axis=1)
    # print (comb_data)
    # mdata = Metadata.detect_from_dataframe(
    #         data=comb_data,
    #         table_name='combined'
    # )
    # mdata.update_column(
    #     column_name='SEX',
    #     sdtype='categorical',
    #     table_name='combined'
    # )
    # mdata.update_column(
    #     column_name='AGE',
    #     sdtype='categorical',
    #     table_name='combined'
    # )
    # synth = TVAESynthesizer (
    #     metadata= mdata,
    #     enforce_min_max_values=True,
    #     epochs=50,
    #     verbose=True,
    # )
    # synth.fit (comb_data)
    # synthetic_data = synth.sample(num_rows=3000)
    # y_synth = synthetic_data[['AGE']]
    # X_synth = synthetic_data.drop(columns=['AGE'])
    # print (X_synth)
    # print (y_synth)

    # Bootstrap training
    print ("starting bootstrap training...")
    pool = mp.Pool(NPOOL)
    input_list = [([df_X_train, df_Y_train, train_cohort,
                    tissue, performance_CUTOFF, norm, agerange, n_bs, split_id] + [seed_list[i]]) for i in range(NUM_BOOTSTRAP)]        
    # input_list = [([X_synth, y_synth, train_cohort,
    #                 tissue, performance_CUTOFF, norm, agerange, n_bs, split_id] + [seed_list[i]]) for i in range(NUM_BOOTSTRAP)]        
    
    coef_list = pool.starmap(Bootstrap_train, input_list)
    pool.close()
    pool.join()
    # print (pls)
    coef_list = pd.concat(coef_list, axis=1).mean(axis=1).abs().sort_values(ascending=False)
    return coef_list        

def Train_all_tissue_aging_model_pls(md_hot_train, df_prot_train,
                                 seed_list, 
                                 performance_CUTOFF, train_cohort,
                                 norm, agerange, n_bs, split_id, NPOOL=15):
    # NUM_BOOTSTRAP = int(n_bs)
    # seed_list = seed_list['BS_Seed']
    # seed_list = seed_list[:NUM_BOOTSTRAP]
    # print(seed_list)
    # final lists for output
    
    with open('gtex/organ_list.dat', 'r') as file:
        tissues = [line.strip() for line in file]

    # Subset to tissue proteins, setup dfX/dfY
    for tissue in tissues:
        dfcoef = Train_tissue_aging_model_pls (
            tissue, md_hot_train, df_prot_train,
            seed_list, 
            performance_CUTOFF, train_cohort,
            norm, agerange, n_bs, split_id, NPOOL=15
        )
        print (dfcoef)
    return dfcoef
  
    
tissue_comp_rate = {
    'liver': 0.05,
    'artery_aorta': 0.09,
    'artery_coronary': 0.02,
    'brain_cortex': 0.01,
    'brain_cerebellum': 0.01,
    'adrenal_gland': 0.01,
    'heart_atrial_appendage': 0.02,
    'pituitary': 0.01,
    'adipose_subcutaneous': 0.1,
    'lung': 0.01,
    'skin_sun_exposed_lower_leg': 0.02,
    'nerve_tibial': 0.01,
    'colon_sigmoid': 0.01,
    'pancreas': 0.09,
    # 'breast_mammary_tissue': 0.01,
    # 'prostate': 0.05,
}

def Bootstrap_train(df_X_train, df_Y_train, train_cohort,
                    tissue, performance_CUTOFF, norm, agerange, n_bs, split_id, seed):
    
    print(f"n_features = {df_X_train.shape[1]}")
    # Setup
    X_train_sample = df_X_train.sample(frac=1, replace=True, random_state=seed).to_numpy()
    Y_train_sample = df_Y_train.sample(frac=1, replace=True, random_state=seed).to_numpy()    
    print("did bootstrap setup... (seed = ", seed, ")")
    
    # PLS
    print("starting PLS regression... (seed = ", seed, ")")
    pls = PLSRegression(max_iter=5000, n_components=int(tissue_comp_rate[tissue]*df_X_train.shape[1]))
    
    # # Defining range of components to try in grid search
    # components = np.rint(np.linspace (0.01*df_X_train.shape[1], 0.15*df_X_train.shape[1], 15)).astype(int)  # limiting to a max of 20 components or fewer
    # print (components) 
    # tuned_parameters = [{'n_components': components}]
    # n_folds = 4
    # print("initialised PLS params setup... (seed = ", seed, ")")
    
    # # Grid Search
    # clf = GridSearchCV(pls, tuned_parameters, cv=n_folds, scoring="neg_mean_absolute_error", refit=False)
    
    # print("gridSearch done... (seed = ", seed, ")")
    # clf.fit(X_train_sample, Y_train_sample)
    # print("gridSearch fitting done... (seed = ", seed, ")")
    
    # gsdf = pd.DataFrame(clf.cv_results_)
    
    # print("Plot and Pick STARTING :(... (seed = ", seed, ")")
    # best_n_components = Plot_and_pick_n_components(gsdf, performance_CUTOFF, plot=False)  # Pick best component count
    # print("Plot and Pick done... (seed = ", seed, ")")
    
    # # Retrain with best parameters
    # pls = PLSRegression(n_components=best_n_components, max_iter=5000)

    pls.fit(X_train_sample, Y_train_sample)
    print("PLS regression retrained... (seed = ", seed, ")")
    
    # SAVE MODEL
    savefp = "gtex/train_splits/train_bs" + n_bs + "_" + split_id + "/data/ml_models/" + train_cohort + "/" + agerange + "/" + norm + "/" + tissue + "/" + train_cohort + "_" + agerange + "_" + norm + "_pls_" + tissue + "_seed" + str(seed) + "_aging_model.pkl"
    pickle.dump(pls, open(savefp, 'wb'))
    
    # SAVE coefficients            
    coef_list = pls.coef_.flatten() # You can adjust this if more details are needed
    coefficients_df = pd.Series(coef_list, index=df_X_train.columns)

    return coefficients_df

def Plot_and_pick_n_components(gsdf, performance_CUTOFF, plot=True):
    """
    This function selects the best number of components based on the performance cutoff and plots the results.
    
    Parameters:
    - gsdf: DataFrame containing the GridSearchCV results
    - performance_CUTOFF: The threshold to determine the acceptable performance range
    - plot: Whether to plot the results (default is True)
    
    Returns:
    - best_n_components: The best number of components based on the performance criteria
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
    x = gsdf.param_n_components.to_numpy()
    y = gsdf.mean_test_score_norm.to_numpy()
    dx = np.diff(x).mean()  # Step size based on the range of n_components
    gsdf["derivative"] = np.gradient(y, dx)
    print("P&P derivative calculated...")

    # Find the number of components with a negative derivative and closest to the performance cutoff
    tmp = gsdf.loc[gsdf.derivative < 0]
    if len(tmp) != 0:
        best_n_components = list(tmp.loc[tmp.mean_test_score_norm_minus_cutoff_abs == np.min(tmp.mean_test_score_norm_minus_cutoff_abs)].param_n_components)[-1]
    else:
        print('No component count with negative derivative, selecting closest to cutoff.')
        tmp2 = gsdf
        best_n_components = list(tmp2.loc[tmp2.mean_test_score_norm_minus_cutoff_abs == np.min(tmp2.mean_test_score_norm_minus_cutoff_abs)].param_n_components)[0]
    
    # Plot the results
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(7, 3))

        # Plot normalized test score vs number of components
        sns.scatterplot(data=gsdf, x="param_n_components", y="mean_test_score_norm", ax=axs[0])
        sns.scatterplot(data=gsdf.loc[gsdf.param_n_components == best_n_components], x="param_n_components", y="mean_test_score_norm", ax=axs[0])

        sns.scatterplot(data=gsdf, x="param_n_components", y="mean_test_score_norm", ax=axs[1])
        sns.scatterplot(data=gsdf.loc[gsdf.param_n_components == best_n_components], x="param_n_components", y="mean_test_score_norm", ax=axs[1])

        axs[0].set_xlim(-0.02, best_n_components + 0.1)
        axs[0].set_ylim(0.8, 1.05)
        axs[0].axvline(best_n_components, color="red", linestyle="--")
        axs[0].axhline(performance_CUTOFF, color="blue", linestyle="--")

        plt.tight_layout()
        plt.show()

    return best_n_components

    
    
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
    
def main():
    agerange="HC"
    performance_CUTOFF=0.95
    norm="Zprot_perf"+str(int(performance_CUTOFF*100))
    train_cohort="gtexV8"

    gene_sort_crit = sys.argv[1]
    n_bs = sys.argv[2]
    split_id = sys.argv[3]
    if gene_sort_crit != '20p' and gene_sort_crit != '1000' and gene_sort_crit != 'deg' and gene_sort_crit != 'AA':
        print ("Invalid gene sort criteria")
        exit (1)
    if int(n_bs) > 500:
        print ("n_bs > 500 not possible")
        exit (1)

    def df_prot_train (tissue):
        return pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/"+tissue+".TRAIN." + split_id + ".tsv", sep='\s+').set_index("Name")
        # return pd.read_csv(filepath_or_buffer="../../../gtex/gtexv8_coronary_artery_TRAIN.tsv", sep='\s+').set_index("Name")

    from md_age_ordering import return_md_hot
    md_hot_train = return_md_hot()

    bs_seed_list = json.load(open("gtex/Bootstrap_and_permutation_500_seed_dict_500.json"))

    #95% performance
    start_time = time.time()
    dfcoef = Train_all_tissue_aging_model_pls(md_hot_train, #meta data dataframe with age and sex (binary) as columns
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


if __name__ == "__main__":
    main()