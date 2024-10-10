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
from sample_attribute_proc import get_samp_attr

gene_sort_crit = sys.argv[1]
n_bs = sys.argv[2]
split_id = sys.argv[3]
regr = sys.argv[4]

if gene_sort_crit != '20p' and gene_sort_crit != '1000' and gene_sort_crit != 'deg':
    print ("Invalid gene sort criteria")
    exit (1)
if int(n_bs) > 500:
    print ("n_bs > 500 not possible")
    exit (1)


class CreateGTExTissueAgeObject:
    # init method or constructor
    def __init__(self, organ, path_bootstrap_seeds='Bootstrap_and_permutation_500_seed_dict_500.json'):
        self.data_and_model_paths = { "path_bootstrap_seeds": path_bootstrap_seeds }
        self.load_data_and_models(organ)
        del self.data_and_model_paths


    def load_data_and_models(self, organ):
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

            # load protein zscore scaler
            fn_protein_scaler = 'gtexV8_HC_based_'+organ+'_gene_zscore_scaler.pkl'
            with open('gtex/train_splits/train_bs' + n_bs + '_' + split_id + '/data/ml_models/gtexV8/HC/'+norm+'/'+organ+"/"+fn_protein_scaler, 'rb') as f_scaler:
                loaded_model = pickle.load(f_scaler)
                models_dict[organ]["prot_scaler"] = loaded_model

            for seed in bootstrap_seeds:
                fn_aging_model = 'gtexV8_HC_'+norm+'_' + regr + '_'+organ+'_seed'+str(seed)+'_aging_model.pkl'
                with open('gtex/train_splits/train_bs' + n_bs + '_' + split_id + '/data/ml_models/gtexV8/HC/'+norm+'/'+organ+"/"+fn_aging_model, 'rb') as f_model:
                    loaded_model = pickle.load(f_model)
                    models_dict[organ]["aging_models"].append(loaded_model)

        # save to object
        self.models_dict = models_dict


    def add_data(self, md_hot, df_prot):
        # to select subset with both sex and protein info
        tmp = pd.concat([md_hot, df_prot], axis=1).dropna()
        if len(tmp) < len(md_hot):
            warnings.warn('Subsetted to samples with both biological sex metadata and gene tpm')
        self.md_hot = md_hot.loc[tmp.index]
        self.df_prot = df_prot.loc[tmp.index]


    def estimate_organ_age(self, organ):
        tpm_scaler = self.models_dict[organ]["prot_scaler"]
        df_input = self.df_prot
        
        smp_attr = get_samp_attr(tissue=organ)
        smp_attr = smp_attr.merge(right = df_input.index.to_series(), how='inner', left_index=True, right_index=True)
        print(smp_attr)

        df_input = pd.concat([self.md_hot[["SEX"]], smp_attr[["SMTSISCH"]], df_input], axis=1)
        df_input[['SMTSISCH']] = df_input[['SMTSISCH']].fillna(df_input[['SMTSISCH']].median())

        df_input = pd.DataFrame(tpm_scaler.transform(df_input),
                                index=df_input.index,
                                columns=df_input.columns)
        # print (df_input)
        predicted_age = self.predict_bootstrap_aggregated_age(df_input, organ)  #aaaaaaaa

        # store results in dataframe
        dfres = self.md_hot.copy()
        dfres["Predicted_Age"] = predicted_age
        dfres = self.calculate_lowess_yhat_and_agegap(dfres, organ)
        # dfres = self.zscore_agegaps(dfres, organ)
        return dfres

    def predict_bootstrap_aggregated_age(self, df_input, organ):
        # predict age across all bootstraps
        predicted_ages_all_seeds = []
        for aging_model in self.models_dict[organ]['aging_models']:
            predicted_ages_seed = aging_model.predict(df_input.to_numpy())
            predicted_ages_all_seeds.append(predicted_ages_seed)
        # take mean of predicted ages
        predicted_ages = np.mean(predicted_ages_all_seeds, axis=0)
        return predicted_ages


    def calculate_lowess_yhat_and_agegap(self, dfres, organ):
        dfres_agegap = dfres.copy()
        # calculate agegap using lowess of predicted vs chronological age from training cohort
        lowess = sm.nonparametric.lowess
        lowess_fit = lowess(dfres_agegap.Predicted_Age.to_numpy(), dfres_agegap.AGE.to_numpy(), frac=2/3, it=5)
        lowess_fit_int = interp1d(lowess_fit[:,0], lowess_fit[:,1], bounds_error=False, kind='linear', fill_value='extrapolate') 
        y_lowess = lowess_fit_int(dfres_agegap.AGE)
        dfres_agegap["yhat_lowess"] = y_lowess
        # dfres_agegap["yhat_lowess"] = age_prediction_lowess(np.array(dfres_agegap.Age))
        if len(dfres_agegap.loc[dfres_agegap.yhat_lowess.isna()]) > 0:
            print("Could not predict lowess yhat in " + str(len(dfres_agegap.loc[dfres_agegap.yhat_lowess.isna()])) + " samples")
            dfres_agegap = dfres_agegap.dropna(subset="yhat_lowess")
        dfres_agegap["AgeGap"] = dfres_agegap["Predicted_Age"] - dfres_agegap["yhat_lowess"]
        return dfres_agegap


from md_age_ordering import return_md_hot
md_hot = return_md_hot()
all_tissue_res = md_hot.copy()

def test_OrganAge (tissue):
    print ("Testing on trained model")
    data = CreateGTExTissueAgeObject(tissue)
    df_prot = pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/" + tissue + ".TEST." + split_id + ".tsv", sep='\s+').set_index("Name")
    df_prot.index.names = ['SUBJID']
    md_hot_tissue = md_hot.merge(right = df_prot.index.to_series(), how='inner', left_index=True, right_index=True)
    print(md_hot_tissue)

    # sample metadata data with Age and Sex_F
    data.add_data(md_hot_tissue, df_prot)
    dfres = data.estimate_organ_age(tissue)
    return dfres


with open('gtex/organ_list.dat', 'r') as file:
    tissues = [line.strip() for line in file]

for tissue in tissues:
    print ("TESTING ON " + tissue)
    res = test_OrganAge(tissue)
    # print (res)
    # print(res["AGE"].describe())

    # print(res["DTHHRDY"].describe())
    toplot = res
    toplot = toplot.sort_values("AGE")
    sns.scatterplot(data=toplot, x="AGE", y="Predicted_Age", 
                    hue="AgeGap", palette='coolwarm', hue_norm=(-12,12), 
                    style="DTHHRDY",  
                    markers={0: "o", 1: "X", 2: "o", 3: "o", 4: "*"})
    plt.xlim (20, 100)
    toplot = toplot.drop_duplicates(subset='AGE')
    x_smooth = np.linspace(toplot.AGE.min(), toplot.AGE.max(), 300)
    quadratic_interp = interp1d(toplot.AGE, toplot.yhat_lowess, kind='quadratic')
    y_smooth = quadratic_interp(x_smooth)
    plt.title("Age gap predictions for " + tissue)
    plt.plot(x_smooth, y_smooth, label='Smoothed line', color='black')
    # plt.show()
    # plt.savefig('gtex/logistic_PTyj_noGS_C10_tstScale_train_bs10.png')
    if regr == "lasso":
        plt.savefig("gtex_outputs/lasso_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + tissue + ".png")
    elif regr == "ridge":
        plt.savefig("gtex_outputs/ridge_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + tissue + ".png")
    elif regr == "elasticnet":
        plt.savefig("gtex_outputs/elasticnet_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + tissue + ".png")
    elif regr == "randomforest":
        plt.savefig("gtex_outputs/randomforest_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + tissue + ".png")
    elif regr == "l1logistic":
        plt.savefig("gtex_outputs/l1logistic_PTyj_f1ma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + tissue + ".png")
    plt.clf()
    # all_tissue_res['p_age_' + tissue] = res["Predicted_Age"]
    all_tissue_res['agegap_' + tissue] = res['AgeGap']
    tissue_res = pd.DataFrame(index=res.index)
    tissue_res['agegap_' + tissue] = res['AgeGap']
    tissue_res['DTHHRDY'] = res['DTHHRDY']
    mse = mean_squared_error(res['AGE'], res['Predicted_Age'])
    r2 = r2_score(res['AGE'], res['Predicted_Age'])
    r2_yhat = r2_score(res['AGE'], res['yhat_lowess'])
    print(f'Mean Squared Error: {mse} = ({mse**0.5})^2')
    print(f'R-squared: {r2}')
    print(f'R-squared with y_hat: {r2_yhat}')
    print() 

exclude_cols = ['AGE', 'SEX', 'DTHHRDY']
subset_cols = [col for col in all_tissue_res.columns if col not in exclude_cols]
all_tissue_res = all_tissue_res.dropna(how='all', subset=subset_cols)

all_tissue_res['non_null_count'] = all_tissue_res.count(axis=1)
all_tissue_res = all_tissue_res.sort_values(by='non_null_count', ascending=False)
all_tissue_res = all_tissue_res.drop(columns=['non_null_count'])


# print (all_tissue_res)
if regr == "lasso":
    all_tissue_res.to_csv("gtex_outputs/lasso_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + ".tsv", sep='\t', index=True)
elif regr == "ridge":
    all_tissue_res.to_csv("gtex_outputs/ridge_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + ".tsv", sep='\t', index=True)
elif regr == "elasticnet":
    all_tissue_res.to_csv("gtex_outputs/elasticnet_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + ".tsv", sep='\t', index=True)
elif regr == "l1logistic":
    all_tissue_res.to_csv("gtex_outputs/l1logistic_PTyj_f1ma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + ".tsv", sep='\t', index=True)
elif regr == "randomforest":
    all_tissue_res.to_csv("gtex_outputs/randomforest_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + ".tsv", sep='\t', index=True)