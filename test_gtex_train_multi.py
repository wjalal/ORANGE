from gtex import GTExTissueAgeBootstrap20700
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def test_OrganAge():
    print ("Testing on trained model")
    data = GTExTissueAgeBootstrap20700.CreateGTExTissueAgeObject()
    df_prot = pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/artery_coronary.TEST.20700.tsv", sep='\s+').set_index("Name")
    # df_prot = pd.read_csv(filepath_or_buffer="../../../gtex/gtexv8_coronary_artery_TEST.tsv", sep='\s+').set_index("Name")
    md_hot = pd.read_csv(filepath_or_buffer="../../../gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS-rangemid_int.txt", sep='\s+').set_index("SUBJID")
    df_prot.index.names = ['SUBJID']
    md_hot = md_hot.merge(right = df_prot.index.to_series(), how='inner', left_index=True, right_index=True)
    print(md_hot)

    # sample metadata data with Age and Sex_F
    data.add_data(md_hot, df_prot)
    # data.normalize(assay_version="v4.1")
    corr = abs(df_prot.corrwith(md_hot["AGE"])).sort_values(ascending=False)
    corr = corr[corr >= 0.5]
    print (corr.size)
    print(corr)

    dfres = data.estimate_organ_ages()
    return dfres

res = test_OrganAge()
print (res)
print(res["AGE"].describe())
toplot = res.loc[res.Organ=="artery_coronary"]
toplot = toplot.sort_values("Predicted_Age")
ageGap = toplot.eval("Predicted_Age - AGE").rename("ageGap")
sns.scatterplot(data=toplot, x="AGE", y="Predicted_Age", 
                hue=ageGap, palette='coolwarm', hue_norm=(-3,3))                
# plt.plot(toplot.Age, toplot.yhat_lowess)
plt.savefig('gtex/lasso_MMS_nms_tstScale_train_bs20_20700.png')
