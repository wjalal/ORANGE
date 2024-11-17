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
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm


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
            p_gt = cond_prob_dthhrdy_gt(df_agegap, i, mu+std*0.5)
            p_lt = cond_prob_dthhrdy_lt(df_agegap, i, mu-std*0.5)
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
            fig, ax = plt.subplots(figsize=(6, 4))
            bin_count = 50
            counts, bin_edges = np.histogram(data_points, bins=bin_count)

            # Custom colormap and normalization
            colors = ['lightgray', 'green', 'yellow', 'orange', 'red']
            cmap = ListedColormap(colors)
            bounds = [0, 1, 2, 3, 4, 5]
            color_norm = BoundaryNorm(bounds, cmap.N)

            for i in range(bin_count):
                bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
                indices = (data_points >= bin_edges[i]) & (data_points < bin_edges[i+1])
                dthhrdy_bin_values = dthhrdy_values[indices]
                sort_key = np.where(dthhrdy_bin_values == 1, -1, dthhrdy_bin_values)
                sorted_indices = np.argsort(sort_key)
                plt.scatter([bin_center] * np.sum(indices), 
                            range(np.sum(indices)),
                            color=cmap(dthhrdy_bin_values[sorted_indices] / max(dthhrdy_values)),  
                            marker='o', s=8)  

            # Plotting the normal curve
            xmin, xmax = plt.xlim()  
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)

            # Define regions under the curve
            region_2 = (x >= mu + 0.5 * std) 
            region_3 = (x <= mu - 0.5 * std) 

            # Fill between the curve and x-axis for each region
            plt.fill_between(x, -1, p * max(counts)/max(p), where=region_2, color='lightgray', alpha=0.25)
            plt.fill_between(x, -1, p * max(counts)/max(p), where=region_3, color='lightgray', alpha=0.25)

            plt.plot(x, p * max(counts)/max(p), 'black', linewidth=0.25)  

            # Set titles and labels
            plt.title(f"{col} distribution")
            plt.xlabel("AgeGap")
            plt.ylabel("Frequency")

            # Set y-axis to integer values
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

            # Adjust plot margins
            plt.subplots_adjust(left=0.08, right=0.99, top=0.999, bottom=0.15)

            # plt.show()
            if regr == "lasso":
                plt.savefig("gtex_outputs/analytics_lasso_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + col + ".png")
            elif regr == "ridge":
                plt.savefig("gtex_outputs/analytics_ridge_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + col + ".png")
            elif regr == "elasticnet":
                plt.savefig("gtex_outputs/analytics_elasticnet_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + col + ".png")
            elif regr == "randomforest":
                plt.savefig("gtex_outputs/analytics_randomforest_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + col + ".png")
            elif regr == "logistic":
                plt.savefig("gtex_outputs/analytics_logistic_PTyj_f1ma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + col + ".png")
            elif regr == "svr":
                plt.savefig("gtex_outputs/analytics_svr_PTyj_f1ma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + col + ".png")
            elif regr == "pls":
                plt.savefig("gtex_outputs/analytics_pls_PTyj_f1ma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + "_" + col + ".png")
            
            # Dummy scatter points for legend entries (represent DTHHRDY values)
            handles = []
            for color, dthhrdy_value in zip(colors, range(len(colors))):
                handle = ax.scatter([], [], color=color, label=f'DTHHRDY {dthhrdy_value}', marker='o', s=20)
                handles.append(handle)

            # Represent Region 2 and Region 3 as triangles for legend
            region2_handle = plt.Line2D([], [], color='coral', marker='v', markersize=10, alpha=0.5,
                                        linestyle='None', label='Region (>= μ + 0.5σ)')
            region3_handle = plt.Line2D([], [], color='green', marker='^', markersize=10, alpha=0.5,
                                        linestyle='None', label='Region (<= μ - 0.5σ)')
            handles.extend([region2_handle, region3_handle])

            # Generate legend
            figlegend = plt.figure(figsize=(3, 2))
            figlegend.legend(handles=handles, loc='center')
            plt.axis('off')

            # Save legend as a separate image
            figlegend.savefig("gtex_outputs/agegap_dist_legend_image.png", bbox_inches='tight')
            plt.close(figlegend)
            plt.close(fig)

            plt.clf()
    return result