import pandas as pd
import math
import sys
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

gene_sort_crit = sys.argv[1]
if gene_sort_crit != '20p' and gene_sort_crit != '1000' and gene_sort_crit != 'deg' and gene_sort_crit != 'AA':
    print ("Invalid args")
    exit (1)
split_id = sys.argv[2]

# organ_list = ["artery_coronary", "muscle_skeletal", "whole_blood", "skin_sun_exposed_lower_leg", "lung", "liver", "heart_left_ventricle", "nerve_tibial", "artery_aorta", "colon_transverse", "colon_sigmoid"]
with open('gtex/organ_list.dat', 'r') as file:
    organ_list = [line.strip() for line in file]

md_hot = pd.read_csv(filepath_or_buffer="../../../gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS-rangemid_int.txt", sep='\s+').set_index("SUBJID")
# Plot the histogram for the 'AGE' column
# plt.figure(figsize=(8,6))
# plt.hist(md_hot['AGE'], bins=20, edgecolor='black')
# plt.title('Distribution of AGE')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()

for organ in organ_list:
    print(organ)
    # with open(f'mbsra/Gtex DEG Code/outputs/deseq_threshold_1/datas/{organ}.csv', 'r') as file:
    #     deg_list = [line.strip() for line in file]
    df_gene = pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/" + organ + ".TRAIN." + split_id + ".tsv", sep='\s+').set_index("Name")
    # print (df_gene)
    df_gene.index.names = ['SUBJID']
    md_hot_organ = md_hot.merge(right = df_gene.index.to_series(), how='inner', left_index=True, right_index=True)

    sampling_strategy = {
        0: 300,  # Class 0 will have 400 samples
        2: 300,  # Class 1 will have 300 samples
        3: 300,  # Class 2 will have 200 samples
        4: 300,
    }

    # SMOTE to generate 4 times more samples
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)  # 4.0 means 4 times the original samples
    X_resampled, y_resampled = smote.fit_resample(df_gene, md_hot_organ['DTHHRDY'])

    # Combine resampled data back into a DataFrame
    df_gene_resampled = pd.DataFrame(X_resampled, columns=df_gene.columns)
    df_gene_resampled['target'] = y_resampled

    print(f"Original samples: {len(df_gene)}, Resampled samples: {len(df_gene_resampled)}")

    df_gene.index.names = ['Name']
    print (df_gene_resampled)
    # df_gene.to_csv("../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/" + organ + ".TRAIN.smote.tsv", sep='\t', index=True)
