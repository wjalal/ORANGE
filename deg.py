import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
import os

def main (threshold, s_organ=None, s = False):
    if s:
        organ_name_list = [s_organ]
    else:   
        with open('gtex/organ_list.dat', 'r') as file:
            organ_name_list = [line.strip() for line in file]

    for organ in organ_name_list:

        counts_df = pd.read_csv(f'dataCSV/gene_reads_2017-06-05_v8_{organ}.csv')
        counts_df = counts_df.drop(columns=['id', 'Description']).set_index('Name')
        counts_df = counts_df.T

        counts_df['SUBJID'] = counts_df.index.str.split('-').str[0]+'-'+counts_df.index.str.split('-').str[1]
        counts_df = counts_df.set_index('SUBJID')

        metadata = pd.read_csv('../../../gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt', sep='\t').set_index('SUBJID')
        metadata = metadata.loc[metadata.index.intersection(counts_df.index)]

        samples_to_keep = ~metadata.AGE.isna()
        counts_df = counts_df.loc[samples_to_keep]
        metadata = metadata.loc[samples_to_keep]

        age_groups = metadata['AGE'].unique().tolist()
        age_groups_sorted = sorted(age_groups, key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))

        significant_genes = []
        upregulated_genes = {}
        downregulated_genes = {}

        os.makedirs (f'deg_outputs/deseq_threshold_{threshold}/plots/', exist_ok=True)
        os.makedirs (f'deg_outputs/deseq_threshold_{threshold}/data/', exist_ok=True)

        for age_group in age_groups_sorted[:]:
            if metadata["AGE"].nunique() <= 1:
                break

            genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= 10]
            counts_df = counts_df[genes_to_keep]
            print(f"counts_df shape: {counts_df.shape}")
            
            inference = DefaultInference(n_cpus=8)
            dds = DeseqDataSet(
                counts=counts_df,
                metadata=metadata,
                design_factors="AGE",
                refit_cooks=True,
                ref_level=["AGE", age_group],
                inference=inference,
                n_cpus=8,
                quiet=True
            )
            dds.deseq2()
            
            for comparison_age_group in age_groups_sorted:
                if comparison_age_group == age_group:
                    continue
                stat_res = DeseqStats(dds, inference=inference, contrast=["AGE", comparison_age_group, age_group])
                stat_res.summary()
                results_df = stat_res.results_df
                significant_df = results_df[results_df['padj'] < 0.05]
                upregulated = significant_df[significant_df['log2FoldChange'] >= threshold*0.01].index.tolist()
                downregulated = significant_df[significant_df['log2FoldChange'] <= -threshold*0.01].index.tolist()
                significant_genes += upregulated + downregulated
                comparison_name = f"{comparison_age_group} vs {age_group}"
                upregulated_genes[comparison_name] = upregulated
                downregulated_genes[comparison_name] = downregulated
            

            counts_df = counts_df[metadata["AGE"] != age_group]
            print(f"counts_df shape: {counts_df.shape}")
            metadata = metadata[metadata["AGE"] != age_group]
            print(f"metadata shape: {metadata.shape}")
            age_groups_sorted.remove(age_group)

        significant_genes = list(set(significant_genes))
        # print(significant_genes)
        pd.DataFrame(significant_genes).to_csv(f'deg_outputs/deseq_threshold_{threshold}/data/{organ}.csv', index=False, header=False)
        len(significant_genes)

        comparison_groups = list(upregulated_genes.keys())
        upregulated_counts = [len(upregulated_genes[group]) for group in comparison_groups]
        downregulated_counts = [len(downregulated_genes[group]) for group in comparison_groups]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(comparison_groups, downregulated_counts, color='lightcoral', label='Downregulated')
        bars2 = ax.bar(comparison_groups, upregulated_counts, bottom=downregulated_counts, color='skyblue', label='Upregulated')
        formatted_labels = [group.replace(' vs ', '\nvs\n') for group in comparison_groups]
        ax.set_xticks(comparison_groups)
        ax.set_xticklabels(formatted_labels)
        ax.set_ylabel('Gene Count')
        ax.set_xlabel('Comparison Groups')
        ax.set_title(f'{organ}({threshold})')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'deg_outputs/deseq_threshold_{threshold}/plots/{organ}.png')
        # plt.show()

if __name__ == "__main__":
    main(1.5)