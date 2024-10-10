import pandas as pd
import math
import sys

# gene_sort_crit = sys.argv[1]
# if gene_sort_crit != '20p' and gene_sort_crit != '1000' and gene_sort_crit != 'deg':
#     print ("Invalid args")
#     exit (1)
    
# organ_list = ["artery_coronary", "muscle_skeletal", "whole_blood", "skin_sun_exposed_lower_leg", "lung", "liver", "heart_left_ventricle", "nerve_tibial", "artery_aorta", "colon_transverse", "colon_sigmoid"]
with open('gtex/organ_list.dat', 'r') as file:
    organ_list = [line.strip() for line in file]

md_hot = pd.read_csv(filepath_or_buffer="../../../gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS-rangemid_int.txt", sep='\s+').set_index("SUBJID")

for organ in organ_list:
    print(organ)
    with open(f'mbsra/Gtex DEG Code/outputs/datas/{organ}.csv', 'r') as file:
        deg_list = [line.strip() for line in file]

    df_gene = pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/" + organ + ".tsv", sep='\s+').set_index("Name")
    # print (df_gene)
    df_gene.index.names = ['SUBJID']
    md_hot_organ = md_hot.merge(right = df_gene.index.to_series(), how='inner', left_index=True, right_index=True)
    # print(md_hot_organ)

    corr = df_gene.corrwith(md_hot_organ["AGE"])
    corr = abs(corr.dropna()).sort_values(ascending=False)
    
    corr = corr[corr > 0.2]
    print (f"corr size = {corr.size}")
    print (f"deg_list size = {len(deg_list)}")

    intersection = corr[corr.isin(deg_list)]
    print (f"intersection size = {intersection.size}")

    # if corr.size > 1000 and gene_sort_crit == '1000':
    #     corr = corr[:1000]
    # print (corr)

    # print (corr.size)
    # df_gene = df_gene[deg_list]
    # df_gene.index.names = ['Name']
    # # print (df_gene)
    # df_gene.to_csv("../../../gtex/proc/proc_data/reduced/corr" + "deg" + "/" + organ + ".tsv", sep='\t', index=True)
