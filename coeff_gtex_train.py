import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline, interp1d
import numpy as np
import sys 
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
import os 
from gene_name import get_gene_symbol
import glob
import shutil

def main (main=False, md_hot_organ = None, tissue=None, delete_model=False):

    remove = False
    gene_sort_crit = sys.argv[1]
    n_bs = sys.argv[2]
    split_id = sys.argv[3]
    regr = sys.argv[4]
    if len(sys.argv) >= 6:
        lpo_sp = "_" + sys.argv[5]
        if len(sys.argv) >= 7:
            if sys.argv[6] == "remove":
                remove = True
    else:
        lpo_sp = ""

    max_gene_count = 50
    if gene_sort_crit != '20p' and gene_sort_crit != '1000' and gene_sort_crit != 'deg' and gene_sort_crit != 'AA':
        print ("Invalid gene sort criteria")
        exit (1)
    if int(n_bs) > 500:
        print ("n_bs > 500 not possible")
        exit (1)


    class CreateGTExTissueAgeObject:
        # init method or constructor
        def __init__(self, organ, idx, path_bootstrap_seeds='Bootstrap_and_permutation_500_seed_dict_500.json'):
            self.data_and_model_paths = { "path_bootstrap_seeds": path_bootstrap_seeds }
            self.load_data_and_models(organ, idx)
            del self.data_and_model_paths


        def load_data_and_models(self, organ, lpo_i):
            # 500 bootstrapped models
            bootstrap_seeds = json.load(resources.open_text("gtex", self.data_and_model_paths["path_bootstrap_seeds"]))["BS_Seed"]
            bootstrap_seeds = bootstrap_seeds[:int(n_bs)]
            models_dict = {}
            # load organ aging models and cognition organ aging models
            model_norms = ["Zprot_perf95"]

            for i in range(len(model_norms)):

                norm = model_norms[i]

                # load all models
                models_dict[organ] = {}
                models_dict[organ]["aging_models"] = []
                models_dict[organ]["aging_model_paths"] = []

                for seed in bootstrap_seeds:
                    fn_aging_model = 'gtexV8_HC_'+norm+'_' + regr + '_'+organ+'_seed'+str(seed)+'_aging_model.pkl'
                    with open('gtex/train_splits/train_bs' + n_bs + '_' + split_id + lpo_i + '/data/ml_models/gtexV8/HC/'+norm+'/'+organ+"/"+fn_aging_model, 'rb') as f_model:
                        loaded_model = pickle.load(f_model)
                        models_dict[organ]["aging_models"].append(loaded_model)
                        models_dict[organ]["aging_model_paths"].append(f_model)

            # save to object
            self.models_dict = models_dict


        def add_data(self, md_hot, df_prot):
            # to select subset with both sex and protein info
            tmp = pd.concat([md_hot, df_prot], axis=1).dropna()
            if len(tmp) < len(md_hot):
                warnings.warn('Subsetted to samples with both biological sex metadata and gene tpm')
            self.md_hot = md_hot.loc[tmp.index]
            self.df_prot = df_prot.loc[tmp.index]


        def get_organ_coeff(self, organ):
            df_input = pd.concat([self.md_hot[["SEX"]], self.df_prot], axis=1)
            # print (df_input)
            # aaaaaa
            coefs = self.get_bootstrap_aggregated_coeff(df_input, organ)  #aaaaaaaa

            # # store results in dataframe
            # dfres = self.md_hot.copy()
            # dfres["Predicted_Age"] = predicted_age


            return coefs

        def get_bootstrap_aggregated_coeff(self, df_input, organ):
            # predict age across all bootstraps
            coefs_all_seeds = []
            for i, aging_model in enumerate(self.models_dict[organ]['aging_models']):
                # Predict age
                # SAVE coefficients 
                if hasattr(aging_model, '_coef_'):
                    coef_list = aging_model._coef_.flatten() # You can adjust this if more details are needed
                elif hasattr(aging_model, 'coef_'):
                    coef_list = aging_model.coef_.flatten() # You can adjust this if more details are needed
                elif hasattr(aging_model, 'dual_coef_') and hasattr(aging_model, 'support_vectors_'): 
                    coef_list = np.dot(aging_model.dual_coef_, aging_model.support_vectors_).flatten()
                coefficients_df = pd.Series(coef_list, index=df_input.columns)
                coefs_all_seeds.append(coefficients_df)
                # return coefficients_df
                # print(coefficients_df)

                if delete_model:
                    # Get the path of the model and delete the file
                    model_path = self.models_dict[organ]["aging_model_paths"][i].name  # Get the path from the file object
                    os.remove(model_path)  # Delete the file
                    print(f"Deleted model file: {model_path}")

            # Take mean of predicted ages
            coefs = pd.concat(coefs_all_seeds, axis=1).mean(axis=1).rename(f"Coef_{organ}").to_frame()
            coefs[f'Weight_{organ}'] = coefs[f'Coef_{organ}'].abs()
            # coefs = coefs.drop(columns=[f'Coef_{organ}'])
            # coefs = coefs.nlargest(n=50, columns="Weight")
            return coefs



    from md_age_ordering import return_md_hot
    if main:
        md_hot = return_md_hot()
    else:
        md_hot = md_hot_organ

    all_tissue_res = md_hot.copy()
    
    def test_coef(tissue, idx):
    # print ("Testing on trained model")
        if main:
            df_prot = pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/" + tissue + ".TEST." + split_id + ".tsv", sep='\s+').set_index("Name")
        else:
            df_prot = pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/" + tissue + ".tsv", sep='\s+').set_index("Name")
        df_prot.index.names = ['SUBJID']
        # print(md_hot)
        if not main:
            df_prot = df_prot.loc[md_hot.index]
        md_hot_tissue = md_hot.merge(right = df_prot.index.to_series(), how='inner', left_index=True, right_index=True)
        # print(md_hot_tissue)
        data = CreateGTExTissueAgeObject(tissue, idx)

        # sample metadata data with Age and Sex_F
        data.add_data(md_hot_tissue, df_prot)
        dfres = data.get_organ_coeff(tissue)
        return dfres

    def test_OrganAge_coef (tissue):
        if lpo_sp != '':
            results = []
            for idx in range(22):
                try:
                    result = test_coef(tissue, "_" + str(idx))
                    results.append(result)
                except FileNotFoundError:
                    print(f"File for index {idx} not found, skipping.")
                    continue  # Skip this iteration if file is not found
            # Sum the weighted results to get the weighted average
            weighted_average_res = pd.concat(results).groupby(level=0).mean()
            
            return weighted_average_res.nlargest(n=max_gene_count,  columns=f"Weight_{tissue}").drop(columns=[f'Weight_{tissue}'])
        else:
            return test_coef(tissue, '').nlargest(n=max_gene_count, columns=f"Weight_{tissue}").drop(columns=[f'Weight_{tissue}'])
    if main:
        with open('gtex/organ_list.dat', 'r') as file:
            tissues = [line.strip() for line in file]

        tissue_coeffs = []

        for tissue in tissues:
            print ("CHECKING COEFF ON " + tissue)
            res = test_OrganAge_coef(tissue)

            res = res.drop(index="SEX", errors='ignore')
            res['Gene_name'] = res.index.map(get_gene_symbol)

            print (res)
            res.to_csv(f"gtex_outputs/coeff{max_gene_count}_{regr}_{tissue}_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + ".tsv", sep='\t', index=True)
            res = res.drop('Gene_name', axis=1)
            tissue_coeffs.append(res)
        #     common_genes = common_genes.merge(df, on=list(common_genes.columns), how='inner')

        # Concatenate indices from all DataFrames into a single Series

        # Merge all DataFrames using outer join to keep all indices and columns
        common_genes = pd.concat(tissue_coeffs, axis=1, join="outer")

        # Sort rows by the count of non-null values in descending order
        common_genes['non_null_count'] = common_genes.notnull().sum(axis=1)
        common_genes = common_genes[common_genes['non_null_count'] >= 3]
        common_genes = common_genes.sort_values(by='non_null_count', ascending=False)

        # Drop the helper column 'non_null_count'
        common_genes = common_genes.drop(columns=['non_null_count'])
        common_genes['Gene_name'] = common_genes.index.map(get_gene_symbol)
        print(common_genes)
        if regr == "lasso":
            common_genes.to_csv(f"gtex_outputs/coeff{max_gene_count}_lasso_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + ".tsv", sep='\t', index=True)
        elif regr == "ridge":
            common_genes.to_csv(f"gtex_outputs/coeff{max_gene_count}_ridge_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + ".tsv", sep='\t', index=True)
        elif regr == "elasticnet":
            common_genes.to_csv(f"gtex_outputs/coeff{max_gene_count}_elasticnet_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + ".tsv", sep='\t', index=True)
        elif regr == "l1logistic":
            common_genes.to_csv(f"gtex_outputs/coeff{max_gene_count}_l1logistic_PTyj_f1ma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + ".tsv", sep='\t', index=True)
        elif regr == "randomforest":
            common_genes.to_csv(f"gtex_outputs/coeff{max_gene_count}_randomforest_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + ".tsv", sep='\t', index=True)
        elif regr == "svr":
            common_genes.to_csv(f"gtex_outputs/coeff{max_gene_count}_svr_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + ".tsv", sep='\t', index=True)
        elif regr == "pls":
            common_genes.to_csv(f"gtex_outputs/coeff{max_gene_count}_pls_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + ".tsv", sep='\t', index=True)

    if remove:
        # Folder pattern to match
        folder_pattern = f"gtex/train_splits/train_bs{n_bs}_{split_id}_*"

        # Find all folders matching the pattern and remove them recursively
        for folder_path in glob.glob(folder_pattern):
            if os.path.isdir(folder_path):  # Check if the path is a directory
                shutil.rmtree(folder_path)
                print(f"Removed directory: {folder_path}")
            else:
                print(f"Skipped non-directory: {folder_path}")

if __name__ == "__main__":
    main(True)