import pandas as pd
import numpy as np

def return_md_hot(curr_ordering = "122100"):
    md_hot_train = pd.read_csv(filepath_or_buffer="../../../gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS-rangemid_int.txt", sep='\s+').set_index("SUBJID")
    md_hot_train['AGE'] = md_hot_train['AGE'].astype(float)
    md_hot_train.loc[md_hot_train['AGE'] == 75, 'AGE'] = 75 + (int(curr_ordering[5])-1) * 10/3
    md_hot_train.loc[md_hot_train['AGE'] == 65, 'AGE'] = 65 + (int(curr_ordering[4])-1) * 10/3
    md_hot_train.loc[md_hot_train['AGE'] == 55, 'AGE'] = 55 + (int(curr_ordering[3])-1) * 10/3
    md_hot_train.loc[md_hot_train['AGE'] == 45, 'AGE'] = 45 + (int(curr_ordering[2])-1) * 10/3
    md_hot_train.loc[md_hot_train['AGE'] == 35, 'AGE'] = 35 + (int(curr_ordering[1])-1) * 10/3
    md_hot_train.loc[md_hot_train['AGE'] == 25, 'AGE'] = 25 + (int(curr_ordering[0])-1) * 10/3
    return md_hot_train