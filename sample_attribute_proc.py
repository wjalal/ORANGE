import pandas as pd
import numpy as np
import json
# pd.set_option('display.max_rows', None)  # Display all rows

def split_at_second_dash(s):
    parts = s.split('-', 2)  # Split into at most 3 parts
    if len(parts) > 2:
        return parts[0] + '-' + parts[1]  # Return the first two parts joined by '-'
    return s  # Return the original string if there are less than 2 dashes

# with open('gtex/organ_list.dat', 'r') as file:
#     tissues = [line.strip() for line in file]
tissue_map = json.load(open("gtex/tissue_name_map.json"))

smp_att = pd.read_csv(filepath_or_buffer="../../../gtex/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep='\t').set_index("SAMPID")
smp_att = smp_att[['SMATSSCR', 'SMTSD', 'SMTSISCH']]

def get_samp_attr (tissue):
    print (tissue)
    smp_att_tissue = smp_att.query(f"SMTSD == '{tissue_map[tissue]}'")
    smp_att_tissue.index = smp_att_tissue.index.map(split_at_second_dash)
    smp_att_tissue = smp_att_tissue.drop(columns=['SMTSD'])
    smp_att_tissue.index.name = 'SUBJID'
    smp_att_tissue = smp_att_tissue.fillna(smp_att_tissue.median())
    # print(smp_att_tissue)
    smp_att_tissue = smp_att_tissue.drop_duplicates()
    # print(smp_att_tissue)

    # missing_count = smp_att_tissue['SMATSSCR'].isnull().sum()
    # print(f"Count of missing values in 'SMATSSCR': {missing_count}")
    # missing_count = smp_att_tissue['SMTSISCH'].isnull().sum()
    # print(f"Count of missing values in 'SMTSISCH': {missing_count}")
    # print()
    return smp_att_tissue