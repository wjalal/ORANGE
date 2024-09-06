from importlib import resources
import pickle
import json
import dill
import pandas as pd
import numpy as np
import warnings

# Class for GTExTissueAge
class CreateGTExTissueAgeObject:

    # init method or constructor
    def __init__(self,
                #  path_version_scale_factors='v4_to_v4.1_scale_dict.json',
                 path_bootstrap_seeds='Bootstrap_and_permutation_500_seed_dict_10.json',
                 ):

        self.data_and_model_paths = {
                                     "path_bootstrap_seeds": path_bootstrap_seeds,
                                    }
        self.load_data_and_models()
        del self.data_and_model_paths


    def load_data_and_models(self):

        # Seqid:scale_factor dictionary
        # version_scale_factors = json.load(resources.open_text("gtex", self.data_and_model_paths["path_version_scale_factors"]))
        # self.version_scale_factors = version_scale_factors

        # # organ:proteinlist dictionary
        # organ_plist_dict1 = json.load(resources.open_text("train.data",
        #                                                  self.data_and_model_paths["path_organ_plist_dict1"]))
        organ_plist_dict1 = ['artery_coronary']
        # organ_plist_dict2 = json.load(resources.open_text("train.data",
        #                                                       self.data_and_model_paths["path_organ_plist_dict2"]))
        # self.organ_plist_dict1 = organ_plist_dict1
        # self.organ_plist_dict2 = organ_plist_dict2

        # 500 bootstrapped models
        bootstrap_seeds = json.load(resources.open_text("gtex", self.data_and_model_paths["path_bootstrap_seeds"]))["BS_Seed"]
        models_dict = {}

        # load organ aging models and cognition organ aging models
        model_norms = ["Zprot_perf95"]
        plist_dicts = [organ_plist_dict1]

        for i in range(len(model_norms)):

            norm = model_norms[i]
            organ_plist_dict = plist_dicts[i]

            # load all models
            for organ in organ_plist_dict:
                models_dict[organ] = {}
                models_dict[organ]["aging_models"] = []

                # load protein zscore scaler
                fn_protein_scaler = 'gtexV8_HC_based_'+organ+'_gene_zscore_scaler.pkl'
                loaded_model = pickle.loads(resources.read_binary('gtex.train_bs_500315.data.ml_models.gtexV8.HC.'+norm+'.' + organ, fn_protein_scaler))
                models_dict[organ]["prot_scaler"] = loaded_model

                # # age gap zscore scaler
                # fn_agegap_scaler = 'KADRC_HC_'+norm+'_lasso_'+organ+'_agegap_zscore_scaler.pkl'
                # loaded_model = pickle.loads(resources.read_binary('organage.data.ml_models.KADRC.'+norm+'.' + organ, fn_agegap_scaler))
                # models_dict[organ]["agegap_scaler"] = loaded_model

                # # age prediction lowess
                # fn_agepred_lowess = 'KADRC_HC_'+norm+'_lasso_' + organ + '_age_prediction_lowess.dill'
                # loaded_model = dill.loads(resources.read_binary('organage.data.ml_models.KADRC.'+norm+'.' + organ, fn_agepred_lowess))
                # models_dict[organ]["age_prediction_lowess"] = loaded_model

                # load all aging models
                for seed in bootstrap_seeds:
                    fn_aging_model = 'gtexV8_HC_'+norm+'_l1logistic_'+organ+'_seed'+str(seed)+'_aging_model.pkl'
                    loaded_model = pickle.loads(resources.read_binary('gtex.train_bs_500315.data.ml_models.gtexV8.HC.'+norm+'.'+ organ, fn_aging_model))
                    models_dict[organ]["aging_models"].append(loaded_model)

        # save to object
        self.models_dict = models_dict

        organ_plist_dict = organ_plist_dict1.copy()
        self.organ_plist_dict = organ_plist_dict
        # del self.organ_plist_dict1



    def add_data(self, md_hot, df_prot):

        # to select subset with both sex and protein info
        tmp = pd.concat([md_hot, df_prot], axis=1).dropna()
        if len(tmp) < len(md_hot):
            warnings.warn('Subsetted to samples with both biological sex metadata and gene tpm')
        self.md_hot = md_hot.loc[tmp.index]
        self.df_prot = df_prot.loc[tmp.index]

        # check that all proteins required by models are in df_prot
        # model_proteins = [prot for prot in self.organ_plist_dict["Organismal"]]
        # for prot in model_proteins:
        #     if not prot in list(df_prot.columns):
        #         warnings.warn('An aging model protein is missing in your data')


    # def normalize(self, assay_version):

    #     # normalizing protein levels
    #     df_prot_norm = self.df_prot.copy()
    #     if assay_version == "v4":
    #         for prot in df_prot_norm.columns:
    #             df_prot_norm[prot] = df_prot_norm[prot] * self.version_scale_factors[prot]
    #     if assay_version == "v4.1":
    #         pass

    #     # warning if protein distribution seems odd
    #     if df_prot_norm.to_numpy().mean() < 500:
    #         warnings.warn("Your protein expression values seem to be logged/transformed. Make sure to input raw protein expression values in RFU units")

    #     # log
    #     df_prot_norm = np.log10(df_prot_norm)
    #     self.df_prot_norm = df_prot_norm


    def estimate_organ_ages(self):
        # Predict organ age and calculate age gaps. store results in dataframe
        resall = []
        for organ in self.organ_plist_dict:

            # only run if all model proteins available
            # nmissing = 0
            # for prot in plist:
            #     if not prot in list(self.df_prot_norm.columns):
            #         nmissing += 1

            # if nmissing==0:
            #     print(organ+"...")
            #     resall.append(dfres)
            # else:
            #     print(organ+" specific proteins missing. Cannot predict "+organ+" age.")
            dfres = self.estimate_one_organ_age(organ) #gfsfgfdgdfgsaaaaaa
            resall.append(dfres)
        dfres_all = pd.concat(resall)
        self.results = dfres_all
        return dfres_all


    def estimate_one_organ_age(self, organ):
        tpm_scaler = self.models_dict[organ]["prot_scaler"]
        df_prot_z = tpm_scaler.transform()
        df_prot_z = pd.DataFrame(tpm_scaler.transform(self.df_prot),
                                index=self.df_prot.index,
                                columns=self.df_prot.columns)
        df_input = pd.concat([self.md_hot[["SEX"]], self.df_prot], axis=1)
        # aaaaaa
        predicted_age = self.predict_bootstrap_aggregated_age(df_input, organ)  #aaaaaaaa

        # store results in dataframe
        dfres = self.md_hot.copy()
        dfres["Predicted_Age"] = predicted_age
        # dfres = self.calculate_lowess_yhat_and_agegap(dfres, organ)
        # dfres = self.zscore_agegaps(dfres, organ)
        dfres["Organ"] = organ
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


    # def calculate_lowess_yhat_and_agegap(self, dfres, organ):
    #     dfres_agegap = dfres.copy()

    #     # calculate agegap using lowess of predicted vs chronological age from training cohort
    #     age_prediction_lowess = self.models_dict[organ]['age_prediction_lowess']
    #     dfres_agegap["yhat_lowess"] = age_prediction_lowess(np.array(dfres_agegap.Age))

    #     if len(dfres_agegap.loc[dfres_agegap.yhat_lowess.isna()]) > 0:
    #         print("Could not predict lowess yhat in " + str(len(dfres_agegap.loc[dfres_agegap.yhat_lowess.isna()])) + " samples")
    #         dfres_agegap = dfres_agegap.dropna(subset="yhat_lowess")

    #     dfres_agegap["AgeGap"] = dfres_agegap["Predicted_Age"] - dfres_agegap["yhat_lowess"]
    #     return dfres_agegap


    # def zscore_agegaps(self, dfres, organ):
    #     dfres_agegap_z = dfres.copy()
    #     # zscore age gaps using scaler defined from training cohort
    #     agegap_scaler = self.models_dict[organ]["agegap_scaler"]
    #     dfres_agegap_z["AgeGap_zscored"] = agegap_scaler.transform(dfres_agegap_z[["AgeGap"]].to_numpy()).flatten()
    #     dfres_agegap_z["AgeGap_zscored"] = dfres_agegap_z["AgeGap_zscored"] - agegap_scaler.transform([[0]]).flatten()[0]
    #     return dfres_agegap_z





