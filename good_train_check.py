from gtex import GTExTissueAgeBootstrap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# df_prot = pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/artery_coronary.TRAIN.tsv", sep='\s+').set_index("Name")
# df_prot = pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/artery_coronary.tsv", sep='\s+').set_index("Name")
df_prot = pd.read_csv(filepath_or_buffer="../../../gtex/gtexv8_coronary_artery_TRAIN.tsv", sep='\s+').set_index("Name")
md_hot = pd.read_csv(filepath_or_buffer="../../../gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS-rangemid.txt", sep='\s+').set_index("SUBJID")
df_prot.index.names = ['SUBJID']
md_hot = md_hot.merge(right = df_prot.index.to_series(), how='inner', left_index=True, right_index=True)
# print(md_hot)
print (md_hot[md_hot.DTHHRDY == 0.0].count())
