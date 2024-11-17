import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import statsmodels.api as sm
import numpy as np
import sys
import json
import warnings
import torch
import torch.nn as nn
import dill
import pickle
from importlib import resources
from sklearn.metrics import mean_squared_error, r2_score

# Ensure the device for PyTorch is set properly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gene_sort_crit = sys.argv[1]
n_bs = sys.argv[2]
split_id = sys.argv[3]
regr = sys.argv[4]

if gene_sort_crit != '20p' and gene_sort_crit != '1000' and gene_sort_crit != 'deg' and gene_sort_crit != 'AA':
    print("Invalid gene sort criteria")
    exit(1)
if int(n_bs) > 500:
    print("n_bs > 500 not possible")
    exit(1)

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First layer
        self.relu = nn.ReLU()                         # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size) # Output layer
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
class CreateGTExTissueAgeObject:
    def __init__(self, organ, path_bootstrap_seeds='Bootstrap_and_permutation_500_seed_dict_500.json'):
        self.data_and_model_paths = {"path_bootstrap_seeds": path_bootstrap_seeds}
        self.models_dict = {}
        self.load_data_and_models(organ)
        # del self.data_and_model_paths

    def load_data_and_models(self, organ, init=True):
        # Load 500 bootstrapped models
        print(organ)
        bootstrap_seeds = json.load(resources.open_text("gtex", self.data_and_model_paths["path_bootstrap_seeds"]))["BS_Seed"]
        bootstrap_seeds = [577]
        self.models_dict[organ] = {}
        self.models_dict[organ]["aging_models"] = []
        
        # load protein zscore scaler
        fn_protein_scaler = 'gtexV8_HC_based_'+organ+'_gene_zscore_scaler.pkl'
        with open('gtex/train_splits/train_bs' + n_bs + '_' + split_id + '/data/ml_models/gtexV8/HC/Zprot_perf95/'+organ+"/"+fn_protein_scaler, 'rb') as f_scaler:
            loaded_model = pickle.load(f_scaler)
            self.models_dict[organ]["prot_scaler"] = loaded_model

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
        }
        
        # Load PyTorch models
        if not init:
            for seed in bootstrap_seeds:
                # Define Neural Network Parameters
                input_size = self.df_prot.shape[1]  # Number of features
                hidden_size = int(tissue_comp_rate[tissue] * (self.df_prot.shape[1]) * 7)  # Set hidden layer size based on tissue component rate
                output_size = 1  # Assuming regression, change if you are doing classification
                
                # Instantiate the model
                model = FeedForwardNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

                fn_aging_model = 'gtexV8_HC_Zprot_perf95_' + regr + '_'+organ+'_seed'+str(seed)+'_aging_model'
                model_path = f'gtex/train_splits/train_bs{n_bs}_{split_id}/data/ml_models/gtexV8/HC/Zprot_perf95/{organ}/{fn_aging_model}.pth'
                # Load the state_dict into the model
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()  # Set the model to evaluation mode
                self.models_dict[organ]["aging_models"].append(model)

    def add_data(self, md_hot, df_prot, tissue):
        # Subset data with biological sex and protein info
        tmp = pd.concat([md_hot, df_prot], axis=1).dropna()
        if len(tmp) < len(md_hot):
            warnings.warn('Subsetted to samples with both biological sex metadata and gene tpm')
        self.md_hot = md_hot.loc[tmp.index]
        self.df_prot = df_prot.loc[tmp.index]

        print (f"hi :\n{self.models_dict}")
        tpm_scaler = self.models_dict[tissue]["prot_scaler"]
        df_prot_z = pd.DataFrame(tpm_scaler.transform(self.df_prot),
                                index=self.df_prot.index,
                                columns=self.df_prot.columns)
        self.df_prot = pd.concat([self.md_hot[["SEX"]], df_prot_z], axis=1)

    def estimate_organ_age(self, organ):
        # df_input = pd.concat([self.md_hot[["SEX"]], self.df_prot], axis=1)
        predicted_age = self.predict_bootstrap_aggregated_age(self.df_prot, organ)
        dfres = self.md_hot.copy()
        dfres["Predicted_Age"] = predicted_age
        dfres = self.calculate_lowess_yhat_and_agegap(dfres, organ)
        return dfres

    def predict_bootstrap_aggregated_age(self, df_input, organ):
        # Predict age across all bootstraps
        predicted_ages_all_seeds = []

        # Convert input to PyTorch Tensor
        df_tensor = torch.tensor(df_input.values, dtype=torch.float32).to(device)

        for model in self.models_dict[organ]['aging_models']:
            with torch.no_grad():
                predicted_ages_seed = model(df_tensor).cpu().numpy()
            predicted_ages_all_seeds.append(predicted_ages_seed)

        # Take the mean of predicted ages
        predicted_ages = np.mean(predicted_ages_all_seeds, axis=0)
        return predicted_ages

    def calculate_lowess_yhat_and_agegap(self, dfres, organ):
        dfres_agegap = dfres.copy()
        lowess = sm.nonparametric.lowess
        lowess_fit = lowess(dfres_agegap.Predicted_Age.to_numpy(), dfres_agegap.AGE.to_numpy(), frac=0.8, it=3)
        lowess_fit_int = interp1d(lowess_fit[:, 0], lowess_fit[:, 1], bounds_error=False, kind='linear', fill_value=(0, 150))
        y_lowess = lowess_fit_int(dfres_agegap.AGE)
        dfres_agegap["yhat_lowess"] = y_lowess
        if len(dfres_agegap.loc[dfres_agegap.yhat_lowess.isna()]) > 0:
            print("Could not predict lowess yhat in " + str(len(dfres_agegap.loc[dfres_agegap.yhat_lowess.isna()])) + " samples")
            dfres_agegap = dfres_agegap.dropna(subset="yhat_lowess")
        dfres_agegap["AgeGap"] = dfres_agegap["Predicted_Age"] - dfres_agegap["yhat_lowess"]
        return dfres_agegap


from md_age_ordering import return_md_hot
md_hot = return_md_hot()

def test_OrganAge(tissue):
    print("Testing on trained model")
    data = CreateGTExTissueAgeObject(tissue)
    df_prot = pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/" + tissue + ".TEST." + split_id + ".tsv", sep='\s+').set_index("Name")
    df_prot.index.names = ['SUBJID']
    md_hot_tissue = md_hot.merge(right=df_prot.index.to_series(), how='inner', left_index=True, right_index=True)

    # Add data to the model
    data.add_data(md_hot_tissue, df_prot, tissue)
    data.load_data_and_models(tissue, False)

    # Estimate organ age using the neural network
    dfres = data.estimate_organ_age(tissue)
    return dfres


# Main loop for testing on all tissues
with open('gtex/organ_list.dat', 'r') as file:
    tissues = [line.strip() for line in file]

all_tissue_res = md_hot.copy()

for tissue in tissues:
    print(f"TESTING ON {tissue}")
    res = test_OrganAge(tissue)
    print(res)
    
    mse = mean_squared_error(res['AGE'], res['Predicted_Age'])
    r2 = r2_score(res['AGE'], res['Predicted_Age'])
    r2_yhat = r2_score(res['AGE'], res['yhat_lowess'])
    print(f'Mean Squared Error: {mse} = ({mse**0.5})^2')
    print(f'R-squared: {r2}')
    print(f'R-squared with y_hat: {r2_yhat}')
    print() 

    # Plotting and saving results
    toplot = res.sort_values("AGE")
    sns.scatterplot(data=toplot, x="AGE", y="Predicted_Age", hue="AgeGap", palette='coolwarm', hue_norm=(-12, 12), style="DTHHRDY", markers={0: "o", 1: "X", 2: "o", 3: "o", 4: "*"})
    plt.xlim(20, 100)

    toplot = toplot.drop_duplicates(subset='AGE')
    x_smooth = np.linspace(toplot.AGE.min(), toplot.AGE.max(), 300)
    quadratic_interp = interp1d(toplot.AGE, toplot.yhat_lowess, kind='quadratic')
    y_smooth = quadratic_interp(x_smooth)
    plt.title(f"Age gap predictions for {tissue}")
    plt.plot(x_smooth, y_smooth, label='Smoothed line', color='black')

    plt.savefig(f"gtex_outputs/{regr}_nma_tstScale_redc{gene_sort_crit}_train_bs{n_bs}_{split_id}_{tissue}.png")
    plt.clf()

    all_tissue_res.loc[res.index, f'agegap_{tissue}'] = res['AgeGap']

# Save results for each regression type
all_tissue_res.to_csv(f"gtex_outputs/{regr}_nma_tstScale_redc{gene_sort_crit}_train_bs{n_bs}_{split_id}.tsv", sep='\t', index=True)
