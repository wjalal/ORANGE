from gtex import GTExTissueAgeBootstrap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def test_OrganAge():
    print ("Testing on trained model")
    data = GTExTissueAgeBootstrap.CreateGTExTissueAgeObject()
    df_prot = pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/artery_coronary.TEST.tsv", sep='\s+').set_index("Name")
    # df_prot = pd.read_csv(filepath_or_buffer="../../../gtex/gtexv8_coronary_artery_TEST.tsv", sep='\s+').set_index("Name")
    # md_hot = pd.read_csv(filepath_or_buffer="../../../gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS-rangemid.txt", sep='\s+').set_index("SUBJID")
    # md_hot = pd.read_csv(filepath_or_buffer="data_with_lower_age.csv", sep='\s+').set_index("SUBJID")
    md_hot= pd.read_csv(filepath_or_buffer=args.file_path, sep='\s+').set_index("SUBJID")
    df_prot.index.names = ['SUBJID']
    md_hot = md_hot.merge(right = df_prot.index.to_series(), how='inner', left_index=True, right_index=True)  
    print(md_hot)

    # sample metadata data with Age and Sex_F
    data.add_data(md_hot, df_prot)
    # data.normalize(assay_version="v4.1")

    dfres = data.estimate_organ_ages()
    return dfres

parser = argparse.ArgumentParser(description="Train GTEx Model")
parser.add_argument("file_path", type=str, help="Path to the CSV file")
parser.add_argument("--output_path", type=str, default="organ_age_results.csv", help="Path to save the output CSV file")
args = parser.parse_args()


res = test_OrganAge()
print (res[["DTHHRDY"]])
res.to_csv(args.output_path, sep='\t')

toplot = res.loc[res.Organ=="artery_coronary"]
toplot = toplot.sort_values("Predicted_Age")
ageGap = toplot.eval("Predicted_Age - AGE").rename("ageGap")
sns.scatterplot(data=toplot, x="AGE", y="Predicted_Age", 
                hue=ageGap, palette='coolwarm', hue_norm=(-3,3))                
# plt.plot(toplot.Age, toplot.yhat_lowess)
plt.show()
