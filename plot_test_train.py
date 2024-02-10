# from organage import OrganAge
from train.OrganAge import CreateOrganAgeObject
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.interpolate import interp1d
from scipy import stats
import numpy as np

md_hot = pd.read_csv("tests/md_hot.csv")
md_hot = md_hot.set_index("ID")

df_prot = pd.read_csv("tests/df_prot.csv")
df_prot = df_prot.set_index("ID")

data = CreateOrganAgeObject()
data.add_data(md_hot, df_prot)
data.normalize(assay_version="v4.1")  #requires "v4" 5k, or "v4.1" 7k
res = data.estimate_organ_ages()

toplot = res.loc[res.Organ=="Heart"]
toplot = toplot.sort_values("Age")
sns.scatterplot(data=toplot, x="Age", y="Predicted_Age", 
                hue="AgeGap_zscored", palette='coolwarm', hue_norm=(-3,3))                
plt.plot(toplot.Age, toplot.yhat_lowess)
plt.show()
