import pandas as pd
import math
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA

gene_sort_crit = sys.argv[1]
if gene_sort_crit != '20p' and gene_sort_crit != '1000' and gene_sort_crit != 'deg' and gene_sort_crit != 'AA':
    print ("Invalid args")
    exit (1)
    
# organ_list = ["artery_coronary", "muscle_skeletal", "whole_blood", "skin_sun_exposed_lower_leg", "lung", "liver", "heart_left_ventricle", "nerve_tibial", "artery_aorta", "colon_transverse", "colon_sigmoid"]
with open('gtex/organ_list.dat', 'r') as file:
    organ_list = [line.strip() for line in file]

md_hot = pd.read_csv(filepath_or_buffer="../../../gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS-rangemid_int.txt", sep='\s+').set_index("SUBJID")

for organ in organ_list:
    print(organ)
    df_gene = pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/" + organ + ".tsv", sep='\s+').set_index("Name")
    # print (df_gene)
    # df_gene.index.names = ['Name']
    # md_hot_organ = md_hot.merge(right = df_gene.index.to_series(), how='inner', left_index=True, right_index=True)
    # print(md_hot_organ)

   # Standardize the data
    # scaled_data = StandardScaler().fit_transform(df_gene)

    # Create a PCA instance: keep 512 components
    pca = PCA(n_components=200)

    # Fit PCA on the scaled data
    principal_components = pca.fit_transform(df_gene)

    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, index=df_gene.index)

    # Save the DataFrame as a CSV file
    pca_df.to_csv("../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/" + organ + ".tsv", sep='\t', index=True)