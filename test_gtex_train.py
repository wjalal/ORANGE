import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline, interp1d
import numpy as np
import sys 
from importlib.resources import files
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


def main (main=False, md_hot_organ = None, tissue=None, delete_model=False, deg_th="", s_organ=None, s = False):

    gene_sort_crit = sys.argv[1]
    n_bs = sys.argv[2]
    split_id = sys.argv[3]
    regr = sys.argv[4]

    if len(sys.argv) >= 6:
        lpo_sp = "_" + sys.argv[5]
    else:
        lpo_sp = ""

    if gene_sort_crit != '20p' and gene_sort_crit != '1000' and gene_sort_crit != 'deg' and gene_sort_crit != 'oh':
        print ("Invalid gene sort criteria")
        exit (1)
    if int(n_bs) > 500:
        print ("n_bs > 500 not possible")
        exit (1)

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

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

    def train_agegap_nn(df, epochs=1000, lr=0.008):
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

    class CreateGTExTissueAgeObject:
        # init method or constructor
        def __init__(self, organ, path_bootstrap_seeds='Bootstrap_and_permutation_500_seed_dict_500.json'):
            self.data_and_model_paths = { "path_bootstrap_seeds": path_bootstrap_seeds }
            self.load_data_and_models(organ)
            del self.data_and_model_paths


        def load_data_and_models(self, organ):
            # 500 bootstrapped models
            bootstrap_seeds = json.load(files("gtex").joinpath(self.data_and_model_paths["path_bootstrap_seeds"]).open("r"))["BS_Seed"]
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
                # load protein zscore scaler
                fn_protein_scaler = 'gtexv10_HC_based_'+organ+'_gene_zscore_scaler.pkl'
                with open('gtex/train_splits/train_bs' + n_bs + '_' + split_id + lpo_sp + '/data/ml_models/gtexv10/HC/'+norm+'/'+organ+"/"+fn_protein_scaler, 'rb') as f_scaler:
                    loaded_model = pickle.load(f_scaler)
                    models_dict[organ]["prot_scaler"] = loaded_model

                for seed in bootstrap_seeds:
                    fn_aging_model = 'gtexv10_HC_'+norm+'_' + regr + '_'+organ+'_seed'+str(seed)+'_aging_model.pkl'
                    with open('gtex/train_splits/train_bs' + n_bs + '_' + split_id + lpo_sp + '/data/ml_models/gtexv10/HC/'+norm+'/'+organ+"/"+fn_aging_model, 'rb') as f_model:
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

            if main:
                # dfres = self.calculate_lowess_yhat_and_agegap(dfres, organ)
                dfres = self.calculate_yhat_and_agegap_with_nn(dfres, organ)
            # dfres = self.zscore_agegaps(dfres, organ)

            return dfres

        def predict_bootstrap_aggregated_age(self, df_input, organ):
            # predict age across all bootstraps
            predicted_ages_all_seeds = []
            for i, aging_model in enumerate(self.models_dict[organ]['aging_models']):
                # Predict age
                predicted_ages_seed = aging_model.predict(df_input.to_numpy())
                predicted_ages_all_seeds.append(predicted_ages_seed)
                
                if delete_model:
                    # Get the path of the model and delete the file
                    model_path = self.models_dict[organ]["aging_model_paths"][i].name  # Get the path from the file object
                    os.remove(model_path)  # Delete the file
                    print(f"Deleted model file: {model_path}")

            # Take mean of predicted ages
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
        

        def calculate_yhat_and_agegap_with_nn(self, dfres, organ):
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



    from md_age_ordering import return_md_hot
    if main:
        md_hot = return_md_hot()
    else:
        md_hot = md_hot_organ

    all_tissue_res = md_hot.copy()

    def test_Organage_regr (tissue):
        # print ("Testing on trained model")
        data = CreateGTExTissueAgeObject(tissue)
        if main:
            df_prot = pd.read_csv(filepath_or_buffer="proc/proc_datav10/reduced/corr" + gene_sort_crit + "/" + tissue + ".TEST." + split_id + ".tsv", sep='\s+').set_index("Name")
        else:
            df_prot = pd.read_csv(filepath_or_buffer="proc/proc_datav10/reduced/corr" + gene_sort_crit + "/" + tissue + ".tsv", sep='\s+').set_index("Name")
        df_prot.index.names = ['SUBJID']
        # print(md_hot)
        if not main:
            df_prot = df_prot.loc[md_hot.index]
        md_hot_tissue = md_hot.merge(right = df_prot.index.to_series(), how='inner', left_index=True, right_index=True)
        # print(md_hot_tissue)

        # sample metadata data with Age and Sex_F
        data.add_data(md_hot_tissue, df_prot)
        dfres = data.estimate_organ_age(tissue)
        return dfres

    def test_OrganAge(tissue):
        # if regr == 'ensemble':
        #     results = []
        #     weights = tissue_ensmb_rate[tissue]  # Get weights for the specified tissue
        #     for idx, r in enumerate(ensemble_regrs):
        #         regr = r
        #         result = test_Organage_regr(tissue)
        #         # Apply the weight to each result
        #         weighted_result = result.mul(weights[idx])
        #         results.append(weighted_result)
        #     regr = 'ensemble'
        #     # Sum the weighted results to get the weighted average
        #     weighted_average_res = pd.concat(results).groupby(level=0).sum()
            
        #     return weighted_average_res
        # else:
        return test_Organage_regr(tissue)
    
    if main:
        if s:
            tissues = [s_organ]
        else:   
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
                plt.savefig("gtex_outputs/lasso_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + "_" + tissue + ".png")
            elif regr == "ridge":
                plt.savefig("gtex_outputs/ridge_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + "_" + tissue + ".png")
            elif regr == "elasticnet":
                plt.savefig("gtex_outputs/elasticnet_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + "_" + tissue + ".png")
            elif regr == "randomforest":
                plt.savefig("gtex_outputs/randomforest_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + "_" + tissue + ".png")
            elif regr == "l1logistic":
                plt.savefig("gtex_outputs/l1logistic_PTyj_f1ma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + "_" + tissue + ".png")
            elif regr == "svr":
                plt.savefig("gtex_outputs/svr_PTyj_f1ma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + "_" + tissue + ".png")
            elif regr == "pls":
                plt.savefig("gtex_outputs/pls_PTyj_f1ma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + lpo_sp + "_" + tissue + ".png")
            plt.clf()
            # all_tissue_res['p_age_' + tissue] = res["Predicted_Age"]
            all_tissue_res.loc[res.index, 'agegap_' + tissue] = res['AgeGap']
            tissue_res = pd.DataFrame(index=res.index)
            tissue_res['agegap_' + tissue] = res['AgeGap']
            tissue_res['DTHHRDY'] = res['DTHHRDY']
            mse = mean_squared_error(res['AGE'], res['Predicted_Age'])
            r2 = r2_score(res['AGE'], res['Predicted_Age'])
            r2_yhat = r2_score(res['AGE'], res['yhat_lowess'])
            print(f'Mean Squared Error: {mse} = ({mse**0.5})^2')
            print(f'R-squared: {r2}')
            print(f'R-squared with y_hat: {r2_yhat}')
            with open(f"gtex_outputs/{regr}_metrics_redc{gene_sort_crit}_train_bs{n_bs}_{split_id}{lpo_sp}.csv", 'a') as f:
                print(f'{split_id},{tissue},{mse},{r2},{deg_th}', file=f)
            print() 

        exclude_cols = ['AGE', 'SEX', 'DTHHRDY']
        subset_cols = [col for col in all_tissue_res.columns if col not in exclude_cols]
        all_tissue_res = all_tissue_res.dropna(how='all', subset=subset_cols)

        all_tissue_res['non_null_count'] = all_tissue_res.count(axis=1)
        all_tissue_res = all_tissue_res.sort_values(by='non_null_count', ascending=False)
        all_tissue_res = all_tissue_res.drop(columns=['non_null_count'])


    # print (all_tissue_res)
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

        return all_tissue_res
    
    else:
        res = test_OrganAge(tissue)
        # print (res)
        return res


if __name__ == "__main__":
    main(True)