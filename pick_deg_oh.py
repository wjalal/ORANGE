import pandas as pd
import math
import sys

def main(s_organ=None, s=False):
    if s:
        organ_list = [s_organ]
    else:
        # Load organ_enriched_genes.csv into a DataFrame
        organ_enriched_df = pd.read_csv('organ_enriched_genes.csv', index_col=None)
        organ_list = organ_enriched_df.columns.tolist()

    from md_age_ordering import return_md_hot
    md_hot = return_md_hot()

    for organ in organ_list:
        print(organ)

        # Get the deg_list from the corresponding column of the DataFrame
        deg_list = organ_enriched_df[organ].dropna().tolist()

        df_gene = pd.read_csv(filepath_or_buffer="proc/proc_datav10/" + organ + ".tsv", sep='\s+').set_index("Name")
        df_gene.index.names = ['SUBJID']
        md_hot_organ = md_hot.merge(right=df_gene.index.to_series(), how='inner', left_index=True, right_index=True)

        df_gene = df_gene[deg_list]
        print(f"deg_list size = {len(deg_list)}")

        df_gene.index.names = ['Name']
        df_gene.to_csv("proc/proc_datav10/reduced/corr" + "oh" + "/" + organ + ".tsv", sep='\t', index=True)

if __name__ == "__main__":
    main()
