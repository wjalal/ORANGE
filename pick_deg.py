import pandas as pd
import math
import sys

# gene_sort_crit = sys.argv[1]
# if gene_sort_crit != '20p' and gene_sort_crit != '1000' and gene_sort_crit != 'deg' and gene_sort_crit != 'AA':
#     print ("Invalid args")
#     exit (1)
    
# organ_list = ["artery_coronary", "muscle_skeletal", "whole_blood", "skin_sun_exposed_lower_leg", "lung", "liver", "heart_left_ventricle", "nerve_tibial", "artery_aorta", "colon_transverse", "colon_sigmoid"]
def main (threshold, s_organ=None, s = False):
    if s:
        organ_list = [s_organ]
    else:   
        with open('gtex/organ_list.dat', 'r') as file:
            organ_list = [line.strip() for line in file]

    from md_age_ordering import return_md_hot
    md_hot = return_md_hot()
    
    for organ in organ_list:
        print(organ)
        with open(f'deg_outputs/deseq_threshold_{threshold}/data/{organ}.csv', 'r') as file:
            deg_list = [line.strip() for line in file]

        df_gene = pd.read_csv(filepath_or_buffer="proc/proc_datav10/" + organ + ".tsv", sep='\s+').set_index("Name")
        # print (df_gene)
        df_gene.index.names = ['SUBJID']
        md_hot_organ = md_hot.merge(right = df_gene.index.to_series(), how='inner', left_index=True, right_index=True)
        # print(md_hot_organ)

        df_gene = df_gene[deg_list]

        # corr = df_gene.corrwith(md_hot_organ["AGE"])
        # corr = abs(corr.dropna()).sort_values(ascending=False)
        
        # corr = corr[corr > 0.20]
        # print (f"corr size = {corr.size}")
        print (f"deg_list size = {len(deg_list)}")
        # # # print (corr.sort_index())

        # intersection = corr.index.intersection(deg_list)
        # print (f"intersection size = {intersection.size}")
        # # print (intersection.to_list())

        # # print(pd.Series(deg_list).sort_values())
        # # if corr.size > 1000 and gene_sort_crit == '1000':
        # #     corr = corr[:1000]
        # # print (corr)

        # # print (corr.size)
        # df_gene = df_gene[intersection.to_list()]
        # df_gene = df_gene[corr.keys().to_list()]
        # df_gene = df_gene[deg_list]
        df_gene.index.names = ['Name']
        # print (df_gene)
        df_gene.to_csv("proc/proc_datav10/reduced/corr" + "deg" + "/" + organ + ".tsv", sep='\t', index=True)


if __name__ == "__main__":
    main(1.5)