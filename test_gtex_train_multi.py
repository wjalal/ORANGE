from gtex import GTExTissueAgeBootstrap28501
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline, interp1d
import numpy as np
import sys 

gene_sort_crit = sys.argv[1]
if gene_sort_crit != '20p' and gene_sort_crit != '1000':
    print ("Invalid args")
    exit (1)
    

def test_OrganAge (tissue):
    print ("Testing on trained model")
    data = GTExTissueAgeBootstrap28501.CreateGTExTissueAgeObject(tissue)
    df_prot = pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/" + tissue + ".TEST.28501.tsv", sep='\s+').set_index("Name")
    # df_prot = pd.read_csv(filepath_or_buffer="../../../gtex/gtexv8_coronary_artery.TEST.28501.tsv", sep='\s+').set_index("Name")
    md_hot = pd.read_csv(filepath_or_buffer="../../../gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS-rangemid_int.txt", sep='\s+').set_index("SUBJID")
    df_prot.index.names = ['SUBJID']
    md_hot = md_hot.merge(right = df_prot.index.to_series(), how='inner', left_index=True, right_index=True)
    # md_hot['DTHHRDY'] = md_hot['DTHHRDY'].astype(str)

    print(md_hot)

    # sample metadata data with Age and Sex_F
    data.add_data(md_hot, df_prot)
    # data.normalize(assay_version="v4.1")
    # corr = abs(df_prot.corrwith(md_hot["AGE"])).sort_values(ascending=False)
    # corr = corr[corr >= 0.5]
    # print (corr.size)
    # print(corr)

    dfres = data.estimate_organ_age(tissue)
    return dfres


with open('gtex/organ_list.dat', 'r') as file:
    tissues = [line.strip() for line in file]

for tissue in tissues:
    print ("TESTING ON " + tissue)
    res = test_OrganAge(tissue)
    print (res)
    print(res["AGE"].describe())

    print(res["DTHHRDY"].describe())
    toplot = res
    toplot = toplot.sort_values("AGE")
    sns.scatterplot(data=toplot, x="AGE", y="Predicted_Age", 
                    hue="AgeGap", palette='coolwarm', hue_norm=(-12,12), 
                    style="DTHHRDY",  
                    markers={0: "o", 1: "X", 2: "o", 3: "o", 4: "*"})
    plt.xlim (20, 100)
    toplot = toplot.drop_duplicates(subset='AGE')
    x_smooth = np.linspace(toplot.AGE.min(), toplot.AGE.max(), 300)
    quadratic_interp = interp1d(toplot.AGE, toplot.yhat_lowess, kind='quadratic')
    y_smooth = quadratic_interp(x_smooth)
    plt.title("Age gap predictions for " + tissue)
    # Plot the smooth line
    plt.plot(x_smooth, y_smooth, label='Smoothed line', color='black')
    # plt.show()
    # plt.savefig('gtex/logistic_PTyj_noGS_C10_tstScale_train_bs20_28501.png')
    plt.savefig("gtex_outputs/lasso_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs20_28501_" + tissue + ".png")
    plt.clf()
