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

def analyse_tissue_agegaps(sp_st, split_id_r1, split_id_r2, n_bs, gene_sort_crit, regr, curr_ordering = "222100"):
    with open('gtex/organ_list.dat', 'r') as file:
        tissues = [line.strip() for line in file]

    N_run = (int(split_id_r2)-int(split_id_r1)+1)
    N_samp = 0
    all_tissue_dth_agegap = {}
    avg_mse = 0
    avg_r2 = 0 
    avg_r2_yhat = 0

    for tissue in tissues:
        all_tissue_dth_agegap[tissue] = []
        for k in range (0,5):
            all_tissue_dth_agegap[tissue].append({
                'p_gt' : 0,
                'p_lt' : 0,
                'p_r' : 0,
                'p_pos' : 0,
                'p_d' : 0,
                'r2' : 0,
                'r2_yhat' : 0,
                'mse' : 0,
            })

    for s in range (int(split_id_r1), int(split_id_r2)+1):
        split_id = sp_st + str(s)
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
                        fn_aging_model = 'gtexV8_HC_'+norm+'_'+regr+'_'+organ+'_seed'+str(seed)+'_aging_model.pkl'
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
                df_prot_z = pd.DataFrame(tpm_scaler.transform(self.df_prot),
                                        index=self.df_prot.index,
                                        columns=self.df_prot.columns)
                df_input = pd.concat([self.md_hot[["SEX"]], df_prot_z], axis=1)
                # print (df_input)
                # aaaaaa
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



        from md_age_ordering import return_md_hot
        md_hot = return_md_hot(curr_ordering=curr_ordering)

        def test_OrganAge (tissue):
            print ("Testing on trained model")
            data = CreateGTExTissueAgeObject(tissue)
            df_prot = pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/" + tissue + ".TEST." + split_id + ".tsv", sep='\s+').set_index("Name")
            df_prot.index.names = ['SUBJID']
            md_hot_tissue = md_hot.merge(right = df_prot.index.to_series(), how='inner', left_index=True, right_index=True)
            # print(md_hot_tissue)

            # sample metadata data with Age and Sex_F
            data.add_data(md_hot_tissue, df_prot)
            dfres = data.estimate_organ_age(tissue)
            return dfres

        for tissue in tissues:
            print ("ANALYSING ON " + tissue)
            res = test_OrganAge(tissue)
        
            tissue_res = pd.DataFrame(index=res.index)
            tissue_res['agegap_' + tissue] = res['AgeGap']
            tissue_res['DTHHRDY'] = res['DTHHRDY']
            result = agegap_dist_analytics (tissue_res, ['agegap_' + tissue], gene_sort_crit, n_bs, split_id, regr, False)
            for k in range (0,5):
                n_pos = len(res.query(f"AgeGap>0 and DTHHRDY=={k}"))
                n_d = len(res.query(f"DTHHRDY=={k}"))
                if n_d == 0:
                    p_pos = 0
                else:
                    p_pos = n_pos/n_d
                all_tissue_dth_agegap[tissue][k]['p_gt'] += result['agegap_' + tissue][k]['p_gt']
                all_tissue_dth_agegap[tissue][k]['p_lt'] += result['agegap_' + tissue][k]['p_lt']
                all_tissue_dth_agegap[tissue][k]['p_r'] += result['agegap_' + tissue][k]['p_r']
                all_tissue_dth_agegap[tissue][k]['p_d'] += result['agegap_' + tissue][k]['p_d']
                all_tissue_dth_agegap[tissue][k]['p_pos'] += p_pos
                all_tissue_dth_agegap[tissue][k]['mse'] += (mean_squared_error(res['AGE'], res['Predicted_Age']) * res.shape[0])
                all_tissue_dth_agegap[tissue][k]['r2'] += (r2_score(res['AGE'], res['Predicted_Age'])  * res.shape[0])
                all_tissue_dth_agegap[tissue][k]['r2_yhat'] +=  (r2_score(res['AGE'], res['yhat_lowess']) * res.shape[0])
            N_samp += res.shape[0]

    for tissue in all_tissue_dth_agegap:
        print("_______________________________")
        print (tissue)
        print("_________________")
        for i in range(0,5):
            p_gt = all_tissue_dth_agegap[tissue][i]['p_gt'] / N_run
            p_lt = all_tissue_dth_agegap[tissue][i]['p_lt'] / N_run
            p_d = all_tissue_dth_agegap[tissue][i]['p_d'] / N_run
            p_r = all_tissue_dth_agegap[tissue][i]['p_r'] / N_run
            mse = all_tissue_dth_agegap[tissue][i]['mse'] / N_run
            r2 = all_tissue_dth_agegap[tissue][i]['r2'] / N_run
            r2_yhat = all_tissue_dth_agegap[tissue][i]['r2_yhat'] / N_run
            if True:
                print (f'avg. p({i}|gt) = {p_gt:.3f}') 
                print (f'avg. p({i}|lt) = {p_lt:.3f}') 
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

                p_pos = all_tissue_dth_agegap[tissue][i]['p_pos']/N_run
                print()
        print(f"avg. p_pos = {p_pos:.3f}")
        print(f"avg. MSE = {mse:.3f} = ({mse**0.5:.3f})^2")
        avg_mse += mse 
        avg_r2 += r2 
        avg_r2_yhat += r2_yhat
        print(f"avg. r2 = {r2:.3f}") 
        print(f"avg. r2_yhat = {r2_yhat:.3f}")
        print()
        print("_______________________________")

    N_samp = N_samp/N_run
    print (f"Average performance of {regr} across organs in {N_run} runs:")
    print (f"Mean Squared Error: {avg_mse/N_samp}")
    print (f"R squared: {avg_r2/N_samp}")
    print (f"R squared with yhat: {avg_r2_yhat/N_samp}")
    return avg_mse/N_samp, avg_r2/N_samp, avg_r2_yhat/N_samp

if __name__ == "__main__":
    gene_sort_crit = sys.argv[1]
    n_bs = sys.argv[2]
    sp_st = sys.argv[3]
    split_id_r1 = sys.argv[4]
    split_id_r2 = sys.argv[5]
    regr = sys.argv[6]

    if gene_sort_crit != '20p' and gene_sort_crit != '1000' and gene_sort_crit != 'deg' and gene_sort_crit != 'AA':
        print ("Invalid gene sort criteria")
        exit (1)
    if int(n_bs) > 500:
        print ("n_bs > 500 not possible")
        exit (1)
    analyse_tissue_agegaps (sp_st=sp_st,
                            split_id_r1=split_id_r1,
                            split_id_r2=split_id_r2,
                            n_bs=n_bs,
                            regr=regr,
                            gene_sort_crit=gene_sort_crit
                            )