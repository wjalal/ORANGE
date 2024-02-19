from gtex import GTExTissueAgeBootstrap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# df_prot = pd.read_csv(filepath_or_buffer="../../../gtex/gtexv8_coronary_artery_TRAIN.tsv", sep='\s+').set_index("Name")
df_prot = pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/artery_coronary.TRAIN.tsv", sep='\s+').set_index("Name")
df_prot_full = pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/artery_coronary.tsv", sep='\s+').set_index("Name")
md_hot = pd.read_csv(filepath_or_buffer="../../../gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS-rangemid.txt", sep='\s+').set_index("SUBJID")
df_prot.index.names = ['SUBJID']
df_prot_full.index.names = ['SUBJID']

md_hot_full = md_hot.merge(right = df_prot_full.index.to_series(), how='inner', left_index=True, right_index=True)
md_hot = md_hot.merge(right = df_prot.index.to_series(), how='inner', left_index=True, right_index=True)

# print(md_hot)

df0 = md_hot_full[md_hot_full.DTHHRDY == 0.0].count().DTHHRDY
df1 = md_hot_full[md_hot_full.DTHHRDY == 1.0].count().DTHHRDY
df2 = md_hot_full[md_hot_full.DTHHRDY == 2.0].count().DTHHRDY
df3 = md_hot_full[md_hot_full.DTHHRDY == 3.0].count().DTHHRDY
df4 = md_hot_full[md_hot_full.DTHHRDY == 4.0].count().DTHHRDY

d0 = md_hot[md_hot.DTHHRDY == 0.0].count().DTHHRDY
d1 = md_hot[md_hot.DTHHRDY == 1.0].count().DTHHRDY
d2 = md_hot[md_hot.DTHHRDY == 2.0].count().DTHHRDY
d3 = md_hot[md_hot.DTHHRDY == 3.0].count().DTHHRDY
d4 = md_hot[md_hot.DTHHRDY == 4.0].count().DTHHRDY

ff0 = md_hot_full[md_hot_full.SEX == 0.0].count().SEX
ff1 = md_hot_full[md_hot_full.SEX == 1.0].count().SEX

f0 = md_hot[md_hot.SEX == 0.0].count().SEX
f1 = md_hot[md_hot.SEX == 1.0].count().SEX

print ("Death stats")
print ("0: ", d0, round(d0/df0*100), sep='\t')
print ("1: ", d1, round(d1/df1*100), sep='\t')
print ("2: ", d2, round(d2/df2*100), sep='\t')
print ("3: ", d3, round(d3/df3*100), sep='\t')
print ("4: ", d4, round(d4/df4*100), sep='\t')

print ("\nSex stats")
print ("F: ", f1, round(f1/ff1*100), sep='\t')
print ("M: ", f0, round(f0/ff0*100), sep='\t')


# for i in range (1,11): 
    # print ("\n\n\nTEST." + str(i) + ".tsv\n\n")
    # df_prot = pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/artery_coronary.TRAIN." + str(i) + ".tsv", sep='\s+').set_index("Name")
    # md_hot = pd.read_csv(filepath_or_buffer="../../../gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS-rangemid.txt", sep='\s+').set_index("SUBJID")
    # df_prot.index.names = ['SUBJID']

    # md_hot_full = md_hot.merge(right = df_prot_full.index.to_series(), how='inner', left_index=True, right_index=True)
    # md_hot = md_hot.merge(right = df_prot.index.to_series(), how='inner', left_index=True, right_index=True)

    # # print(md_hot)

    # df0 = md_hot_full[md_hot_full.DTHHRDY == 0.0].count().DTHHRDY
    # df1 = md_hot_full[md_hot_full.DTHHRDY == 1.0].count().DTHHRDY
    # df2 = md_hot_full[md_hot_full.DTHHRDY == 2.0].count().DTHHRDY
    # df3 = md_hot_full[md_hot_full.DTHHRDY == 3.0].count().DTHHRDY
    # df4 = md_hot_full[md_hot_full.DTHHRDY == 4.0].count().DTHHRDY

    # d0 = md_hot[md_hot.DTHHRDY == 0.0].count().DTHHRDY
    # d1 = md_hot[md_hot.DTHHRDY == 1.0].count().DTHHRDY
    # d2 = md_hot[md_hot.DTHHRDY == 2.0].count().DTHHRDY
    # d3 = md_hot[md_hot.DTHHRDY == 3.0].count().DTHHRDY
    # d4 = md_hot[md_hot.DTHHRDY == 4.0].count().DTHHRDY

    # ff0 = md_hot_full[md_hot_full.SEX == 0.0].count().SEX
    # ff1 = md_hot_full[md_hot_full.SEX == 1.0].count().SEX

    # f0 = md_hot[md_hot.SEX == 0.0].count().SEX
    # f1 = md_hot[md_hot.SEX == 1.0].count().SEX

    # print ("Death stats")
    # print ("0: ", d0, round(d0/df0*100), sep='\t')
    # print ("1: ", d1, round(d1/df1*100), sep='\t')
    # print ("2: ", d2, round(d2/df2*100), sep='\t')
    # print ("3: ", d3, round(d3/df3*100), sep='\t')
    # print ("4: ", d4, round(d4/df4*100), sep='\t')

    # print ("\nSex stats")
    # print ("F: ", f1, round(f1/ff1*100), sep='\t')
    # print ("M: ", f0, round(f0/ff0*100), sep='\t')