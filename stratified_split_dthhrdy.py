import pandas as pd
import math
import sys
from sklearn.model_selection import train_test_split

gene_sort_crit = sys.argv[1]
rand_seed = sys.argv[2]
if gene_sort_crit != '20p' and gene_sort_crit != '1000':
    print ("Invalid args")
    exit (1)

with open('gtex/organ_list.dat', 'r') as file:
    organ_list = [line.strip() for line in file]

md_hot = pd.read_csv(filepath_or_buffer="../../../gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS-rangemid_int.txt", sep='\s+').set_index("SUBJID")
md_hot['DTHHRDY'] = md_hot['DTHHRDY'].fillna(0)

for organ in organ_list:
    print(organ)
    df_gene = pd.read_csv("../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/" + organ + ".tsv", sep='\s+').set_index("Name")
    # print (df_gene)
    df_gene.index.names = ['SUBJID']
    md_hot_organ = md_hot.merge(right = df_gene.index.to_series(), how='inner', left_index=True, right_index=True)
    df_gene['DTHHRDY'] = md_hot_organ['DTHHRDY']

    df_gene_test_dthhrdy_1 = df_gene[df_gene['DTHHRDY'] == 1]
    df_gene_remain = df_gene[df_gene['DTHHRDY'] != 1]

    df_gene_train, df_gene_test = train_test_split(df_gene_remain, test_size=0.2, stratify=df_gene_remain['DTHHRDY'], random_state=int(rand_seed))
    
    df_gene_test = pd.concat([df_gene_test_dthhrdy_1, df_gene_test])
    
    df_gene_train = df_gene_train.drop(columns=['DTHHRDY'])
    df_gene_test = df_gene_test.drop(columns=['DTHHRDY'])
    df_gene_train.index.names = ['Name']
    df_gene_test.index.names = ['Name']
    print(df_gene_train.shape)
    print(df_gene_test.shape)


    df_gene_train.to_csv("../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/" + organ + ".TRAIN.cl1sp" + rand_seed + ".tsv", sep='\t', index=True)
    df_gene_test.to_csv("../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/" + organ + ".TEST.cl1sp" + rand_seed + ".tsv", sep='\t', index=True)