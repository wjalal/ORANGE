import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
import numpy as np
import sys 
from importlib import resources
import warnings
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import norm
from matplotlib.colors import ListedColormap, BoundaryNorm
from CondProbDthHrdy import *


def agegap_dist_analytics (df, cols, gene_sort_crit, n_bs, split_id, regr, plot):
    result = {}
    for col in cols:
        print ("=====================")
        print (col)
        result[col] = []
        print ("=====================\n")
        data_points = df[[col]].values.flatten()
        data_points = data_points[~np.isnan(data_points)]
        dthhrdy_values = df['DTHHRDY'].repeat(df[[col]].count(axis=1)).values

        # Fit a normal distribution to the data
        mu, std = norm.fit(data_points)

        df_agegap = pd.DataFrame()
        df_agegap["agegap"] = data_points
        df_agegap["dthhrdy"] = dthhrdy_values

        for i in range(0,5):
            result[col].append({})
            p_gt = cond_prob_dthhrdy_gt(df_agegap, i, mu+std)
            p_lt = cond_prob_dthhrdy_lt(df_agegap, i, mu-std)
            # if round(p_gt,5) == 0 and round(p_lt,5) == 0:
            #     r = 1
            #     r_i = 1
            # elif p_gt == 0:
            #     r = 0
            #     r_i = float('inf')
            # elif p_lt == 0:
            #     r = float('inf')
            #     r_i = 0
            # else:
            #     r = p_gt/p_lt
            #     r_i = 1/r
            # if r > 1: 
            #     print (f'Extreme positive agers are {r:.3f} times more likely to have died with dthhrdy={i} than extreme negative agers ')
            # elif r < 1:
            #     print (f'Extreme NEGATIVE agers are {r_i:.3f} times more likely to have died with dthhrdy={i} than extreme positive agers ')
            # else:
            #     print(f'Extreme negative and positive agers are equally likely to have died with dthhrdy={i}')
            # cond_prob_dthhrdy_gt(df_agegap, i, mu+std*1.5)
            # cond_prob_dthhrdy_lt(df_agegap, i, mu-std*1.5)
            p_r = cond_prob_dthhrdy_range(df_agegap, i, mu-std/5, mu+std/5)
            p_d =  prob_dthhrdy(df_agegap, i)
            # cond_prob_gt_dthhrdy(df_agegap, mu+std, i)
            # cond_prob_lt_dthhrdy(df_agegap, mu-std, i)
            # cond_prob_gt_dthhrdy(df_agegap, mu+std*1.5, i)
            # cond_prob_lt_dthhrdy(df_agegap, mu-std*1.5, i)
            # cond_prob_range_dthhrdy(df_agegap, mu-std/5, mu+std/5, i)
            result[col][i]['p_gt'] = p_gt
            result[col][i]['p_lt'] = p_lt
            result[col][i]['p_r'] = p_r
            result[col][i]['p_d'] = p_d
            print()

        if plot == True:
            bin_count = 50
            counts, bin_edges = np.histogram(data_points, bins=bin_count)
            # cmap = plt.get_cmap('coolwarm')

            # Define your custom colormap and boundary normalization
            colors = ['lightgray', 'green', 'yellow', 'orange', 'red']
            cmap = ListedColormap(colors)
            bounds = [0, 1, 2, 3, 4, 5]  # Boundaries between each DTHHRDY value (add 5 to cover the range)norm = BoundaryNorm(bounds, cmap.N)

            for i in range(bin_count):
                # Get the x-coordinate for the center of the bin
                bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
                # Get the indices for the dots in this bin
                indices = (data_points >= bin_edges[i]) & (data_points < bin_edges[i+1])
                # Scatter the points vertically for this bin with color based on 'DTHHRDY'
                # Sort dthhrdy_values for this bin in ascending order
                dthhrdy_bin_values = dthhrdy_values[indices]
                sorted_indices = np.argsort(dthhrdy_bin_values)
                plt.scatter([bin_center] * np.sum(indices), 
                            range(np.sum(indices)),
                            color=cmap(dthhrdy_bin_values[sorted_indices] / max(dthhrdy_values)),  # Normalize DTHHRDY values
                            marker='o', s=2)  # s is the size of the markers

            # plotting normal curve
            xmin, xmax = plt.xlim()  
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p * max(counts)/max(p), 'black', linewidth=0.5)  

            # Highlight specific regions under the curve
            # region_1 = (x >= mu - 1 * std) & (x <= mu + 1 * std)  # 1 std region
            region_2 = (x >= mu + 1 * std) 
            region_3 = (x <= mu - 1 * std) 

            # Fill between the curve and the x-axis for each region
            # plt.fill_between(x, 0, p, where=region_1, color='lightblue', alpha=0.5)
            plt.fill_between(x, 0, p, where=region_2, color='coral', alpha=0.5)
            plt.fill_between(x, 0, p, where=region_3, color='green', alpha=0.5)

            plt.title("All organ ageGap distribution")
            plt.xlabel("AgeGap")
            plt.ylabel("Frequency")
            # plt.show()
            if regr == "lasso":
                plt.savefig("gtex_outputs/analytics_lasso_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + col + ".png")
            elif regr == "logistic":
                plt.savefig("gtex_outputs/analytics_logistic_PTyj_f1ma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + col + ".png")
            plt.clf()
    return result