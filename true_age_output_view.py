import pandas as pd
import numpy as np
import json
import sys

gene_sort_crit = sys.argv[1]
n_bs = sys.argv[2]
split_id_r = sys.argv[3]
split_id = "cl1sp" + str(split_id_r)
regr = sys.argv[4]

if gene_sort_crit != '20p' and gene_sort_crit != '1000' and gene_sort_crit != 'deg':
    print ("Invalid gene sort criteria")
    exit (1)
if int(n_bs) > 500:
    print ("n_bs > 500 not possible")
    exit (1)

with open('gtex/organ_list.dat', 'r') as file:
    organ_list = [line.strip() for line in file]

for organ in organ_list:
    df = pd.read_csv(filepath_or_buffer=f'gtex_outputs/{regr}_corr{gene_sort_crit}_bs{n_bs}_{split_id}_age_guess.csv', 
                    sep=',',
                    dtype={'ordering': 'object'}).set_index("ordering")

    df['mse_r2_q'] = df['mse'] / df['r2']


    # df.index = df.index.str.replace('0', '-').str.replace('1', '.').str.replace('2', '+')
    # df = df.sort_values(by=['r2', 'mse'], ascending=[False, True])  # Sort 'mse' in ascending and 'r2' in descending order
    # df = df.sort_values(by=['mse', 'r2'], ascending=[True, False])  # Sort 'mse' in ascending and 'r2' in descending order
    df = df.sort_values(by=['mse_r2_q'], ascending=[True])

    print(df)

