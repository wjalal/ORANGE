import pandas as pd
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
from agegap_analytics import *
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Create a lighter version by blending with white
def lighten_cmap(cmap, factor=0.7):
    colors = cmap(np.linspace(0, 1, 256))
    lightened_colors = (1 - factor) * colors + factor * np.ones_like(colors)
    return mcolors.LinearSegmentedColormap.from_list('lightened_coolwarm', lightened_colors)


gene_sort_crit = sys.argv[1]
n_bs = sys.argv[2]
sp_st = sys.argv[3]
split_id_r1 = sys.argv[4]
split_id_r2 = sys.argv[5]
regr = sys.argv[6]
if len(sys.argv) >= 8:
    agg = sys.argv[7]  #lpo
else:
    agg=""

if gene_sort_crit != '20p' and gene_sort_crit != '1000' and gene_sort_crit != 'deg' and gene_sort_crit != 'AA':
    print ("Invalid gene sort criteria")
    exit (1)
if int(n_bs) > 500:
    print ("n_bs > 500 not possible")
    exit (1)

tissues = [
    "liver",
    "artery_aorta",
    "artery_coronary",
    "brain_cortex",
    "brain_cerebellum",
    # "adrenal_gland",
    "heart_atrial_appendage",
    # "pituitary",
    "adipose_subcutaneous",
    "lung",
    "skin_sun_exposed_lower_leg",
    "nerve_tibial",
    "colon_sigmoid",
    "pancreas",
#    "breast_mammary_tissue",
#    "prostate",
]
# cols = ['max_agegap', 'min_agegap']
cols = ['max_agegap']

# all_tissue_dth_agegap = {}
# for col in cols:
#     all_tissue_dth_agegap[col] = []
#     for k in range (0,5):
#         all_tissue_dth_agegap[col].append({
#             'p_gt' : 0,
#             'p_lt' : 0,
#             'p_r' : 0,
#             'p_d' : 0,
#         })
        
agegap_matrices = []
pheno_data = None

pd.set_option('display.min_rows', 25)


for s in range (int(split_id_r1), int(split_id_r2)+1):
    split_id = sp_st + str(s)
    if regr == "lasso":
        all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/lasso_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + agg + ".tsv", sep='\t').set_index("SUBJID")
    elif regr == "ridge":
        all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/ridge_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id +  agg + ".tsv", sep='\t').set_index("SUBJID")
    elif regr == "elasticnet":
        all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/elasticnet_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + agg + ".tsv", sep='\t').set_index("SUBJID")
    elif regr == "logistic":
        all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/logistic_PTyj_f1ma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + agg + ".tsv", sep='\t').set_index("SUBJID")
    elif regr == "randomforest":
        all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/randomforest_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + agg + ".tsv", sep='\t').set_index("SUBJID")
    elif regr == "svr":
        all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/svr_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + agg + ".tsv", sep='\t').set_index("SUBJID")
    elif regr == "pls":
        all_tissue_res = pd.read_csv(filepath_or_buffer="gtex_outputs/pls_PTyj_nma_tstScale_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id + agg + ".tsv", sep='\t').set_index("SUBJID")

    exclude_cols = ['AGE', 'SEX', 'DTHHRDY']
    pheno_data = all_tissue_res[exclude_cols]
    subset_cols = [col for col in all_tissue_res.columns if col not in exclude_cols and col[7:] in tissues]
    # print (subset_cols)
    all_tissue_res = all_tissue_res.drop(columns=[col for col in  all_tissue_res.columns if col[7:] not in tissues and col not in exclude_cols])

        # Set the minimum number of non-null columns required per row
    # min_non_null_columns = int(len(tissues)/2)
    min_non_null_columns = 5
    # Filter rows with at least 6 non-null values
    all_tissue_res = all_tissue_res.dropna(thresh=min_non_null_columns+len(exclude_cols))
    # print (all_tissue_res.shape)
    scaler = StandardScaler()
    # scaler = MinMaxScaler(feature_range=(-7,7))
    all_tissue_res[subset_cols] = scaler.fit_transform(all_tissue_res[subset_cols])
    print (all_tissue_res)
    agegap_matrices.append(all_tissue_res[subset_cols])

# print(agegap_matrices)

# Concatenate along a new axis and compute cell-wise mean and standard deviation
concat_df = pd.concat(agegap_matrices, axis=0, keys=range(len(agegap_matrices)))
concat_df.to_csv("gtex_outputs/idk.tsv", sep='\t', index=True)
print(concat_df)
# Calculate the mean and standard deviation, ignoring NaNs
mean_df = concat_df.groupby(level=1).mean()
mean_df['non_null_count'] = mean_df.count(axis=1)
mean_df = mean_df.sort_values(by='non_null_count', ascending=False)
mean_df = mean_df.drop(columns=['non_null_count'])

std_df = concat_df.groupby(level=1).std()

# # Display the resulting DataFrames
print("Mean DataFrame:\n", mean_df)
print("\nStandard Deviation DataFrame:\n", std_df)

# Calculate the average of the standard deviations for each column
avg_std_per_column = std_df.mean(axis=0)

# Calculate the overall average of the column averages from `mean_df`
overall_avg_of_std = np.nanmean(std_df.values)

print("Average of standard deviations per column:\n", avg_std_per_column)
print("\nOverall average of standard deviations:", overall_avg_of_std)

rows_with_extreme_values = mean_df[mean_df[(mean_df > 2) | (mean_df < -2)].any(axis=1)]
print(rows_with_extreme_values)

with open("extreme_agers.csv", "w") as f:
    for index in rows_with_extreme_values.index:
        f.write(f"{index}\n")

# Initialize an empty DataFrame for extreme_denoised
extreme_denoised = pd.DataFrame()
ageotype_avg_agegaps = pd.DataFrame()

# Iterate over each tissue and save indices with extreme agegap values for each one
for tissue in tissues:
    column_name = f"agegap_{tissue}"
    
    # Check if the column exists in mean_df (to avoid errors if a column is missing)
    if column_name in mean_df.columns:
        # Filter rows where values in the specific tissue column are < -2 or > 2
        extreme_rows = mean_df[(mean_df[column_name] < -2) | (mean_df[column_name] > 2)]
        
        # Perform a set union with extreme_denoised
        extreme_denoised = pd.concat([extreme_denoised, extreme_rows]).drop_duplicates()

        # Save the indices of these extreme rows to a CSV file for each tissue
        filename = f"gtex_outputs/{tissue}_extreme_agers.csv"
        extreme_rows.index.to_series().to_csv(filename, header=False, index=False)
        print(f"Saved extreme agers for {tissue} to {filename}")

        # Calculate the mean of all rows in extreme_rows and insert into ageotype_avg_agegaps
        if not extreme_rows.empty:  # Check to avoid errors when extreme_rows is empty
            mean_row = extreme_rows.abs().mean(axis=0)  # Row-wise mean
            mean_row.name = f"{tissue}"    # Set index name for the row
            ageotype_avg_agegaps = pd.concat([ageotype_avg_agegaps, mean_row.to_frame().T])
    else:
        print(f"Column {column_name} not found in mean_df.")

extreme_denoised.columns = extreme_denoised.columns.str.slice(7)
extreme_denoised = extreme_denoised.rename(columns={
    "heart_atrial_appendage": "heart", 
    "skin_sun_exposed_lower_leg": "skin_sun_exposed"
})
print (f"extreme denoised: {extreme_denoised.shape}")
ageotype_avg_agegaps.columns = ageotype_avg_agegaps.columns.str.slice(7)
ageotype_avg_agegaps = ageotype_avg_agegaps.rename(columns={
    "heart_atrial_appendage": "heart", 
    "skin_sun_exposed_lower_leg": "skin_sun_exp"
})
ageotype_avg_agegaps = ageotype_avg_agegaps.rename({
    "heart_atrial_appendage": "heart", 
    "skin_sun_exposed_lower_leg": "skin_sun_exp"
})


# Heatmap of extreme_denoised
plt.figure(figsize=(4, 4))
# Draw heatmap
ax = sns.heatmap(
    abs(extreme_denoised),
    cmap=lighten_cmap(plt.cm.Reds, 0.2),
    annot=False,
    cbar_kws={"shrink": 0.9},
    xticklabels=True,
    yticklabels=False,  # Remove row labels for squeezing
    square=True
)
ax.set_aspect(0.09)  # Compress the height (smaller values = flatter rows)
# Customize axes
plt.xticks(fontsize=6, rotation=30, ha="right")
plt.yticks(fontsize=6)
plt.xlabel("Absolute z-scored tissue age-gap", fontsize=8)
plt.ylabel("Extreme agers (grouped by tissue ageotype)", fontsize=7)
# Save the heatmap
plt.tight_layout()
plt.savefig(f"gtex_outputs/extreme_denoised_heatmap_{regr}_{split_id_r1}-{split_id_r2}.png", dpi=300, bbox_inches="tight")
# plt.show()


non_extreme_denoised = mean_df.loc[~mean_df.index.isin(rows_with_extreme_values.index)]
non_extreme_denoised.columns = non_extreme_denoised.columns.str.slice(7)
non_extreme_denoised = non_extreme_denoised.rename(columns={
    "heart_atrial_appendage": "heart", 
    "skin_sun_exposed_lower_leg": "skin_sun_exposed"
})
print (f"non-extreme denoised: {non_extreme_denoised.shape}")
full_denoised = pd.concat([non_extreme_denoised, extreme_denoised])
print (full_denoised)

# Heatmap of full_denoised
import matplotlib.patches as patches
from matplotlib.path import Path

# Plot the heatmap
plt.figure(figsize=(4, 4))
ax = sns.heatmap(
    full_denoised,
    cmap=lighten_cmap(plt.cm.bwr, 0.1),
    annot=False,
    cbar_kws={"shrink": 0.7},
    xticklabels=True,
    yticklabels=False,  # Suppress default row labels
    square=True
)
ax.set_aspect(0.02)  # Compress the height (smaller values = flatter rows)

def add_curly_brace(ax, y_start, y_end, x_position, label, bulge=0.5, fontsize=6):
    """Add a curly brace to indicate a section in the heatmap."""
    # Define the path for the curly brace
    verts = [
        (x_position, y_start),  # Start of the brace
        (x_position - bulge, (y_start + y_end) / 2),  # Middle curve
        (x_position, y_end)  # End of the brace
    ]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    path = Path(verts, codes)

    # Add the brace as a PathPatch
    brace = patches.PathPatch(path, linewidth=0.5, edgecolor="black", capstyle='round', facecolor="none")
    ax.add_patch(brace)

    # Add text for the label
    ax.text(
        x_position - bulge, (y_start + y_end) / 2, label,
        rotation=90, fontsize=fontsize, va='center', ha='right'
    )

# Adjust the x-axis to avoid clipping
ax.set_xlim(-2, full_denoised.shape[1])  # Add some space on the left

# Add brace for "Non-extreme agers"
add_curly_brace(ax, y_start=10, y_end=non_extreme_denoised.shape[0]-10, x_position=-0.5, bulge=0.9, label="Non-extreme agers")

# Add brace for "Extreme-agers (grouped by tissue ageotype)"
add_curly_brace(ax, y_start=non_extreme_denoised.shape[0]+10, y_end=full_denoised.shape[0]-10, x_position=-0.5, label="Extreme agers\n(grouped by \ntissue ageotype)")

# Customize x-axis and y-axis
plt.xticks(fontsize=6, rotation=30, ha="right")
plt.xlabel("z-scored tissue age-gaps", fontsize=7)
plt.ylabel("", fontsize=8)  # Leave blank or update as needed

# Save the heatmap
plt.tight_layout()
plt.savefig(f"gtex_outputs/full_denoised_heatmap_{regr}_{split_id_r1}-{split_id_r2}.png", dpi=300, bbox_inches="tight")


# print(ageotype_avg_agegaps)
plt.figure(figsize=(4, 4))
# Draw heatmap
ax = sns.heatmap(
    abs(ageotype_avg_agegaps),
    cmap=lighten_cmap(plt.cm.Reds, 0.3),
    annot=False,
    cbar_kws={"shrink": 0.65},
    xticklabels=True,
    yticklabels=True,  # Remove row labels for squeezing
    square=True
)
# Customize axes
plt.xticks(fontsize=4, rotation=45, ha="right")
plt.yticks(fontsize=4, rotation=45, va="top")
plt.xlabel("Avg. tissue age-gaps", fontsize=8)
plt.ylabel("Subject ageotypes", fontsize=8)
# Save the heatmap
plt.tight_layout()
plt.savefig(f"gtex_outputs/ageotype_avg_agegap__{regr}_{split_id_r1}-{split_id_r2}.png", dpi=300, bbox_inches="tight")


rows_with_one_extreme_value = mean_df[((mean_df > 2) | (mean_df < -2)).sum(axis=1) == 1]
print(rows_with_one_extreme_value)

# Save the original index of rows_with_extreme_values
original_index = rows_with_extreme_values.index
multi_extreme_rows = rows_with_extreme_values.merge(rows_with_one_extreme_value, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
multi_extreme_rows = multi_extreme_rows[((multi_extreme_rows > 2) | (multi_extreme_rows < -2)).sum(axis=1) >= 3]
# Reassign the original index
multi_extreme_rows.index = original_index[multi_extreme_rows.index]
print(multi_extreme_rows)
multi_extreme_rows.index.to_series().to_csv(f"gtex_outputs/multi_extreme_agers.csv", header=False, index=False)

mdf = mean_df.copy()
mdf.columns = mdf.columns.str.slice(7)
# Step 1: Calculate correlation matrix with absolute values
corr_matrix = mdf.corr(min_periods=1)
# Step 2: Calculate the pairwise count of non-null values
count_matrix = mdf.notnull().T.dot(mdf.notnull())
print(count_matrix)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

# Step 3: Weight correlations by count and calculate the weighted average
# (excluding self-correlations with np.diag_indices)
weighted_corr_sum = (corr_matrix * count_matrix).values.sum() - np.trace(corr_matrix.values * count_matrix.values)
total_weights = count_matrix.values.sum() - np.trace(count_matrix.values)
average_weighted_corr = weighted_corr_sum / total_weights

print(f"Average weighted correlation: {average_weighted_corr}")

plt.figure(figsize=(4, 4))
ax = sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    cmap=lighten_cmap(plt.cm.bwr, factor=0.4),
    vmin=-1,
    vmax=1,
    square=True,  # Ensures square cells
    cbar_kws={"shrink": 0.6},  # Shrink the colorbar if needed
    annot_kws={"fontsize": 3},  # Small font for annotations
    # linewidths=0.5,  # Thin grid lines for better clarity
)
plt.xticks(rotation=30, ha='right', fontsize=4)  # Small rotated x-axis labels
plt.yticks(fontsize=4)  # Small y-axis labels
# Adjust colorbar font size
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=4)  # Set colorbar tick font size
plt.tight_layout()  # Ensures no padding
plt.savefig(
    f"gtex_outputs/agegap_{regr}_correlation_heatmap_cmn{split_id_r1}-{split_id_r2}.png",
    format="png",
    dpi=300,
    bbox_inches="tight"
)




# Step 1: Filter for rows with at least 9 non-null columns and exclude extreme rows
non_extreme_rows = mean_df.loc[(~mean_df.index.isin(rows_with_extreme_values.index)) & (mean_df.notnull().sum(axis=1) >= 9)]
# Step 2: Calculate the row-wise standard deviation and sort in ascending order
non_extreme_rows = non_extreme_rows.assign(stddev=non_extreme_rows.std(axis=1))
non_extreme_rows = non_extreme_rows.sort_values(by='stddev', ascending=True).drop(columns='stddev')
print ("non extreme")
print(non_extreme_rows)

# Step 2: Sample 5 rows from the filtered DataFrame
sampled_rows = non_extreme_rows.head(5)  # Random seed for reproducibility
sampled_rows.columns = sampled_rows.columns.str.slice(7)

print ("multi-sel")
# Step 2: Sample 5 rows from the filtered DataFrame
# Step 2: Calculate the row-wise standard deviation and sort in ascending order
# mx_rows = rows_with_extreme_values.loc[rows_with_extreme_values.notnull().sum(axis=1) >= 7]
# mx_rows = mx_rows.assign(stddev=mx_rows.std(axis=1))
# mx_rows = mx_rows.sort_values(by='stddev', ascending=False).drop(columns='stddev')
# sampled_mx_rows = mx_rows.sample(n=50, random_state=1)  # Random seed for reproducibility

# sampled_mx_rows = multi_extreme_rows.head(6)

sampled_mx_rows = mean_df.loc[['GTEX-1E2YA', 'GTEX-11EMC', 'GTEX-145MO', 'GTEX-1RAZQ', 'GTEX-1A8FM']]

sampled_mx_rows.columns = sampled_mx_rows.columns.str.slice(7)
sampled_mx_rows = sampled_mx_rows.rename({
    'GTEX-1E2YA' : 'GTEX-1E2YA\n(brain ager)', 
    'GTEX-11EMC' : 'GTEX-11EMC\n(lung ager)', 
    'GTEX-145MO' : 'GTEX-145MO\n(pancreas ager)', 
    'GTEX-1RAZQ' : 'GTEX-1RAZQ\n(multi-organ ager)', 
    'GTEX-1A8FM' : 'GTEX-1A8FM\n(multi-organ ager)',
})


# Create a subplot with 1 row and 2 columns (shared axes for y)
fig, ax = plt.subplots(1, 2, figsize=(6, 4), sharey=True)

# Define the colormap
cmap = plt.get_cmap('tab20')

# Add horizontal dotted lines at y=0, y=2, and y=-2 for both subplots
for a in ax:
    a.axhline(y=0, color='lightgray', linestyle='--', linewidth=0.7)  # y=0 line
    a.axhline(y=2, color='lightcoral', linestyle='--', linewidth=0.7)  # y=2 line
    a.axhline(y=-2, color='lightblue', linestyle='--', linewidth=0.7)  # y=-2 line

# Plot for non-extreme rows
for i, column in enumerate(sampled_rows.columns):
    ax[0].plot(
        sampled_rows.index, 
        sampled_rows[column], 
        color=cmap(i),
        linestyle='None', 
        marker='o', 
        markersize=10,              # Large marker size
        markeredgewidth=1,          # Width of the black edge
        markeredgecolor='black',    # Black stroke around each marker
        label=column
    )

# Plot for extreme rows (multi-extreme rows)
for i, column in enumerate(sampled_mx_rows.columns):
    ax[1].plot(
        sampled_mx_rows.index, 
        sampled_mx_rows[column], 
        color=cmap(i),
        linestyle='None', 
        marker='o', 
        markersize=10,              # Large marker size
        markeredgewidth=1,          # Width of the black edge
        markeredgecolor='black',    # Black stroke around each marker
        label=column
    )

# Set axis labels and titles for both subplots
ax[0].set_ylabel("Age-gaps (z-scored)")
ax[0].set_xlabel("Non-extreme-aging subjects")
ax[1].set_xlabel("Extreme-aging subjects")

# Adjust y-label position to align vertically
ax[0].xaxis.set_label_coords(0.5, -0.28)  # Adjust y-label position for ax[0]
ax[1].xaxis.set_label_coords(0.5, -0.28)  # Adjust y-label position for ax[1]

# Rotate x-axis labels by 30 degrees for better readability
for a in ax:
    a.set_xticklabels(a.get_xticklabels(), rotation=30, ha='right', fontsize=6)
    a.set_ylim(top=4, bottom=-4)

# Add a legend to the figure (shared between both plots)
# Plot the first row to collect the legend handles without duplicating them
handles, labels = ax[0].get_legend_handles_labels()

# Display the legend using the collected handles and labels, closer to the plot
fig.legend(
    handles, labels, 
    title="Tissues", 
    bbox_to_anchor=(1.02, 0.6), 
    loc='center left',
    labelspacing=0.85,
    fontsize='xx-small',  # Reduced legend text size
    markerscale=0.6       # Reduced marker size in the legend
)

# Adjust layout to reduce space between subplots and legend
plt.subplots_adjust(wspace=0.1, right=0.85)  # Adjust the width space and right margin

# Adjust layout and save the combined figure
plt.tight_layout()
plt.savefig(f"gtex_outputs/{regr}_cmn_{split_id_r1}_{split_id_r2}_combined_ager_examples.png", format="png", dpi=300, bbox_inches="tight")



mean_df[['AGE', 'SEX', 'DTHHRDY']] = pheno_data[['AGE', 'SEX', 'DTHHRDY']]
if regr == "pls":
    mean_df.to_csv("gtex_outputs/bs_agegaps_zscpred_pls_redc" + gene_sort_crit + "_train_bs" + n_bs + "_" + split_id_r1 + "-" + split_id_r2 + agg + ".tsv", sep='\t', index=True)
