import pandas as pd
import numpy as np 
import sys 
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def main():
    gene_sort_crit = sys.argv[1]
    rand_seed = sys.argv[2]
    if gene_sort_crit != '20p' and gene_sort_crit != '1000' and gene_sort_crit != 'deg' and gene_sort_crit != 'AA':
        print ("Invalid args")
        exit (1)

    with open('gtex/organ_list.dat', 'r') as file:
        organ_name_list = [line.strip() for line in file]

    from md_age_ordering import return_md_hot
    md_hot = return_md_hot()
    md_hot['DTHHRDY'] = md_hot['DTHHRDY'].fillna(0)

    organ_list = []

    for organ in organ_name_list:
        obj = {
            'name' : organ,
            'df' : pd.read_csv("../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/" + organ + ".tsv", sep='\s+').set_index("Name")
        }
        organ_list.append(obj)

    # print(organ_list)
    index_counter = Counter()

    for organ in organ_list:
        index_counter.update(organ['df'].index)

    # Filter indices that appear in at least `n` dataframes
    n = 5  # Define the minimum number of dataframes the index should appear in
    common_indices = [index for index, count in index_counter.items() if count >= n]
    print (len(common_indices))

    union_indices = set()

    for organ in organ_list:
        # Find the intersection of the organ's dataframe with the common indices
        df_gene = organ['df']
        df_gene.index.names = ['SUBJID']
        
        md_hot_organ = md_hot.merge(right = df_gene.index.to_series(), how='inner', left_index=True, right_index=True)
        df_gene['DTHHRDY'] = md_hot_organ['DTHHRDY']
        df_gene['AGE'] = md_hot_organ['AGE']

        organ_df_intersection = df_gene.loc[df_gene.index.intersection(common_indices)]

        df_gene_dthhrdy_1 = df_gene[df_gene['DTHHRDY'] == 1]
        df_gene_test = pd.concat([df_gene_dthhrdy_1, organ_df_intersection])
        df_gene_test = df_gene_test.drop_duplicates()

        df_gene_remain = df_gene.loc[df_gene.index.difference(df_gene_test.index)]
        intr_cov = df_gene_test.shape[0]/df_gene.shape[0]
        # print(f"Organ: {organ['name']}\n Intersection coverage: {intr_cov:.3f} {df_gene_remain.shape[0]/df_gene.shape[0]:.3f}")

        if intr_cov > 0.21:
            df_gene_test_dthhrdy_1 = df_gene_test[df_gene_test['DTHHRDY'] == 1]
            r_1 = df_gene_test_dthhrdy_1.shape[0] / df_gene_test.shape[0] * intr_cov
            # Remove rows where class size is less than 2
            class_counts = df_gene_test['DTHHRDY'].value_counts().drop(1.0)
            valid_classes = class_counts[class_counts >= 2].index
            # print (class_counts)
            # Keep only the rows with valid classes
            df_gene_test_filtered = df_gene_test[df_gene_test['DTHHRDY'].isin(valid_classes)]

            extra_frac = 1 - (0.2 - r_1)/(intr_cov - r_1)
            _, df_test_return = train_test_split(df_gene_test_filtered, stratify=df_gene_test_filtered['DTHHRDY'], test_size=extra_frac, random_state=int(rand_seed))
            
            df_gene_test = df_gene_test.loc[df_gene_test.index.difference(df_test_return.index)]
            df_gene_train = pd.concat([df_gene_remain, df_test_return])
            df_gene_train = df_gene_train.drop_duplicates()
        else :
            df_gene_train = df_gene_remain

        print(f"Organ: {organ['name']}\n test/train: {df_gene_test.shape[0]/df_gene.shape[0]:.3f} {df_gene_train.shape[0]/df_gene.shape[0]:.3f}")
        print(df_gene_train['DTHHRDY'].unique())
        print(df_gene_test['DTHHRDY'].unique())
        print(df_gene_train.index.intersection(df_gene_test.index))

        print(df_gene_train['AGE'].value_counts(normalize=True))
        print(df_gene_test['AGE'].value_counts(normalize=True))

        df_gene_train = df_gene_train.drop(columns=['DTHHRDY','AGE'])
        df_gene_test = df_gene_test.drop(columns=['DTHHRDY','AGE'])
        df_gene_train.index.names = ['Name']
        df_gene_test.index.names = ['Name']
        # print(df_gene_train.shape)
        # print(df_gene_test.shape)
        union_indices = union_indices.union(df_gene_test.index)

        df_gene_train.to_csv("../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/" + organ['name'] + ".TRAIN.cmn" + rand_seed + ".tsv", sep='\t', index=True)
        df_gene_test.to_csv("../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/" + organ['name'] + ".TEST.cmn" + rand_seed + ".tsv", sep='\t', index=True)

    union_series = pd.Series(list(union_indices))
    print (len(union_series))
    print (union_series)

if __name__ == "__main__":
    main()