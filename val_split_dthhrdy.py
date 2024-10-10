import pandas as pd
import math
import sys

gene_sort_crit = sys.argv[1]
if gene_sort_crit != '20p' and gene_sort_crit != '1000' and gene_sort_crit != 'deg':
    print ("Invalid args")
    exit (1)

with open('gtex/organ_list.dat', 'r') as file:
    organ_list = [line.strip() for line in file]

md_hot = pd.read_csv(filepath_or_buffer="../../../gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS-rangemid_int.txt", sep='\s+').set_index("SUBJID")

for organ in organ_list:
    print(organ)
    df_gene = pd.read_csv("../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/" + organ + ".tsv", sep='\s+').set_index("Name")
    # print (df_gene)
    df_gene.index.names = ['SUBJID']
    md_hot_organ = md_hot.merge(right = df_gene.index.to_series(), how='inner', left_index=True, right_index=True)
    df_gene['DTHHRDY'] = md_hot_organ['DTHHRDY']
    # print(md_hot_organ)
    # print(df_gene)

    train_dthhrdy = [0, 3]

    df_gene_train = df_gene[df_gene['DTHHRDY'].isin(train_dthhrdy)]  
    df_gene_test = df_gene[~df_gene['DTHHRDY'].isin(train_dthhrdy)] 
    df_gene_train = df_gene_train.drop(columns=['DTHHRDY'])
    df_gene_test = df_gene_test.drop(columns=['DTHHRDY'])
    df_gene_train.index.names = ['Name']
    df_gene_test.index.names = ['Name']
    # print(df_gene_train)
    # print(df_gene_test)
    if df_gene_train.shape[0] < df_gene_test.shape[0]:
        print("WARNING: train set is smaller")

    df_gene_train.to_csv("../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/" + organ + ".TRAIN.stsp" + ''.join(map(str, train_dthhrdy)) + ".tsv", sep='\t', index=True)
    df_gene_test.to_csv("../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/" + organ + ".TEST.stsp" + ''.join(map(str, train_dthhrdy)) + ".tsv", sep='\t', index=True)