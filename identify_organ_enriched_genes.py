import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

organs = ["adipose-subcutaneous","artery-aorta","artery-coronary","brain-cerebellum","brain-cortex","colon-sigmoid","heart-atrial-appendage","liver","lung","nerve-tibial","pancreas","skin-sun-exposed-lower-leg"]

# Generate metadata and gene data in PyDeseq2 format.... (Only run for the first time)

# organ_files = [f"./dataCSV/gene_reads_v10_{organ.replace('-', '_')}.csv" for organ in organs]
# organ_data = {f"{organ}": pd.read_csv(file, index_col=0).drop(columns=["Description"], errors="ignore") for organ, file in zip(organs, organ_files)}
# for organ, data in organ_data.items():
#     print(f"{organ}: {data.columns.tolist()}")

# combined_data_list = []
# metadata_list = []
# for organ, data in organ_data.items():
#     organ_metadata = pd.DataFrame({
#         "subjid": [f"{organ}-{subjid}" for subjid in data.columns],
#         "organ": organ
#     })
#     metadata_list.append(organ_metadata)
#     transposed_data = data.T
#     transposed_data.index = [f"{organ}-{subjid}" for subjid in transposed_data.index]
#     combined_data_list.append(transposed_data)
# combined_data = pd.concat(combined_data_list)
# metadata = pd.concat(metadata_list).reset_index(drop=True)
# metadata.set_index("subjid", inplace=True)

# combined_data.to_csv("./combined_gene_data.csv")
# metadata.to_csv("./metadata.csv")

# print("Combined Data Shape:", combined_data.shape)
# print("Metadata Shape:", metadata.shape)




metadata = pd.read_csv("./metadata.csv", index_col=0)
combined_data = pd.read_csv("./combined_gene_data.csv", index_col=0)
common_index = metadata.index.intersection(combined_data.index)
metadata = metadata.loc[common_index]
combined_data = combined_data.loc[common_index]
combined_data.fillna(0, inplace=True)
combined_data = combined_data.astype(int)
metadata = metadata.loc[common_index]
print("Combined Data Shape:", combined_data.shape)
print("Metadata Shape:", metadata.shape)

organ_enriched_genes = {}
for organ in organs:
    genes_to_keep = combined_data.columns[combined_data.sum(axis=0) >= 10]
    combined_data = combined_data[genes_to_keep]

    inference = DefaultInference(n_cpus=8)
    dds = DeseqDataSet(
            counts=combined_data,
            metadata=metadata,
            design_factors="organ",
            refit_cooks=True,
            ref_level=["organ", organ],
            inference=inference,
            n_cpus=8,
            quiet=True
        )
    dds.deseq2()

    organ_specific_genes = None
    organ_upregulated_genes = None
    organ_downregulated_genes = None

    for comp_organ in organs:
        if comp_organ == organ:
            continue

        stat_res = DeseqStats(dds, inference=inference, contrast=["organ", comp_organ, organ])
        stat_res.summary()
        results_df = stat_res.results_df
        significant_df = results_df[results_df['padj'] < 0.05]
        upregulated = significant_df[(significant_df['log2FoldChange'] / 2) >= 1].index.tolist()
        downregulated = significant_df[(significant_df['log2FoldChange'] / 2)  <= -1].index.tolist()
        if organ_upregulated_genes is None:
            organ_upregulated_genes = set(upregulated)
        else:
            organ_upregulated_genes = organ_upregulated_genes.intersection(upregulated)
        if organ_downregulated_genes is None:
            organ_downregulated_genes = set(downregulated)
        else:
            organ_downregulated_genes = organ_downregulated_genes.intersection(downregulated)
    
    organ_enriched_genes[organ] = {
        "upregulated": list(organ_upregulated_genes) if organ_upregulated_genes else [],
        "downregulated": list(organ_downregulated_genes) if organ_downregulated_genes else []
    }

    genes_to_remove = organ_enriched_genes[organ]["upregulated"] + organ_enriched_genes[organ]["downregulated"]
    combined_data = combined_data.drop(columns=genes_to_remove, errors="ignore")

print(organ_enriched_genes)

import json
# Save dictionary to JSON file
with open("organ_enriched_genes.json", "w") as json_file:
    json.dump(organ_enriched_genes, json_file, indent=4)  # `indent` makes the JSON more readable

print("Dictionary saved to organ_enriched_genes.json")

json_file_path = './organ_enriched_genes.json'  
with open(json_file_path, 'r') as file:
    data = json.load(file)
organ_genes = {}
for organ, values in data.items():
    organ_name = organ.replace("-", "_")
    combined_genes = values.get("upregulated", []) + values.get("downregulated", [])
    organ_genes[organ_name] = combined_genes
max_len = max(len(genes) for genes in organ_genes.values())
for organ in organ_genes:
    organ_genes[organ].extend([None] * (max_len - len(organ_genes[organ])))
df = pd.DataFrame(organ_genes)
df.to_csv("./organ_enriched_genes.csv", index=False)


