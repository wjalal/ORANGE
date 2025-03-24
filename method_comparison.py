import pandas as pd
import os
import sys


import matplotlib.pyplot as plt
import numpy as np

# Arrays for algorithms and feature selection methods
# algorithms = ["pls", "elasticnet", "lasso", "randomforest", "svr"]
# feature_selection_methods = ["20p", "deg", "oh"]

algos = {
    "elasticnet": ["20", plt.cm.Oranges], 
    "pls": ["20", plt.cm.Blues],
    "svr": ["20", plt.cm.Greys],
    "randomforest": ["1", plt.cm.Greens],
}

running_times = {
    "liver" : {
        "pls" : 21,
        "elasticnet": 123
    }
}

feature_selection_methods = ["deg", "20p"]
split_id = sys.argv[1]
# Initialize a dictionary to store the results
compiled_data = []

# Base directory where the files are stored
base_dir = "gtex_outputs"  # Change this to your directory

# Iterate through all combinations of algorithms and feature selection methods
for algo in algos.keys():
    for method in feature_selection_methods:
        # Generate the filename based on the current combination
        filename = f"{algo}_metrics_redc{method}_train_bs{algos[algo][0]}_{split_id}.csv"
        file_path = os.path.join(base_dir, filename)

        # Check if the file exists
        if os.path.exists(file_path):
            # Load the CSV file
            data = pd.read_csv(file_path, header=None, index_col=False, names=["experiment_id", "tissue", "mse", "r2"],sep=",")
            
            # Add algorithm and feature selection method as columns
            data["algorithm"] = algo
            data["feature_selection"] = method
            
            # Append to the compiled data
            compiled_data.append(data)
        else:
            print(f"File not found: {filename}")

# Concatenate all the data into a single dataframe
compiled_df = pd.concat(compiled_data, ignore_index=True)
compiled_df['tissue'] = compiled_df['tissue'].replace("skin_sun_exposed_lower_leg", "skin_sun_exp")
compiled_df['tissue'] = compiled_df['tissue'].replace("adipose_subcutaneous", "adipose_subc")
compiled_df['tissue'] = compiled_df['tissue'].replace("heart_atrial_appendage", "heart_atr_app")



# Print the compiled dataframe and matrix
print("Compiled DataFrame:")
print(compiled_df)

# Save the compiled matrix to a CSV file if needed
# result_matrix.to_csv("compiled_matrix.csv")

# Helper function to sample colors from a colormap
def get_colors_from_cmap(cmap, num_colors):
    return [cmap(((num_colors-1-i)*0.75+1) / (num_colors*1.2)) for i in range(num_colors)]

def plot_grouped_bar_chart(df, split_id):
    # Aggregate data to remove duplicates
    # Specify the desired algorithm order
    algo_order = algos.keys()
    
    # Convert the 'algorithm' column to a categorical type with the specified order
    df['algorithm'] = pd.Categorical(df['algorithm'], categories=algo_order, ordered=True)

    df_aggregated = df.groupby(['tissue', 'algorithm', 'feature_selection']).mean().reset_index()
    print(df_aggregated)

    # Calculate average RMSE and R^2 across methods for sorting
    avg_metrics = df_aggregated.groupby('tissue')[['mse', 'r2']].mean()
    avg_metrics['rmse'] = avg_metrics['mse'] ** 0.5
    avg_metrics = avg_metrics.sort_values(by=['rmse', 'r2'], ascending=[True, False])
    print(avg_metrics)

    # Reorder the tissues based on average values
    tissues_sorted_rmse = avg_metrics.index.tolist()
    tissues_sorted_r2 = avg_metrics.sort_values(by='r2', ascending=False).index.tolist()

    # Unique algorithms and feature selection methods
    algorithms = df_aggregated['algorithm'].unique()
    feature_selections = df_aggregated['feature_selection'].unique()

    # Define custom color palette
    color_palette = {
        algo: get_colors_from_cmap(algos[algo][1], len(feature_selections))
        for algo in algorithms
    }

    bar_width = 0.1

    # Plot RMSE
    fig, ax = plt.subplots(figsize=(21, 3))
    x_positions = np.arange(len(tissues_sorted_rmse))

    for algo_idx, algo in enumerate(algorithms):
        for method_idx, method in enumerate(feature_selections):
            # Subset data for the current combination
            subset = df_aggregated[(df_aggregated['algorithm'] == algo) &
                                   (df_aggregated['feature_selection'] == method)]

            # Align subset with sorted tissues
            subset = subset.set_index('tissue').reindex(tissues_sorted_rmse)

            # Bar positions
            offset = (algo_idx * len(feature_selections) + method_idx*0.65) * bar_width
            positions = x_positions + offset

            # Get color for the current algorithm and feature selection
            color = color_palette[algo][method_idx]

            # Plot RMSE bars
            ax.bar(positions, subset['mse'] ** 0.5, width=bar_width, color=color, label=f'{algo}-{method} RMSE')

    # Configure RMSE plot
    ax.set_xticks(x_positions + (len(algorithms) * len(feature_selections) * bar_width) / 2)
    ax.set_xticklabels(tissues_sorted_rmse)
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE Grouped by Tissue, Algorithm, and Feature Selection')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"gtex_outputs/method_comparison_rmse_{split_id}.png")
    plt.close(fig)

    # Plot R^2
    fig, ax = plt.subplots(figsize=(21, 3))
    x_positions = np.arange(len(tissues_sorted_r2))

    for algo_idx, algo in enumerate(algorithms):
        for method_idx, method in enumerate(feature_selections):
            # Subset data for the current combination
            subset = df_aggregated[(df_aggregated['algorithm'] == algo) &
                                   (df_aggregated['feature_selection'] == method)]

            # Align subset with sorted tissues
            subset = subset.set_index('tissue').reindex(tissues_sorted_r2)

            # Bar positions
            offset = (algo_idx * len(feature_selections) + method_idx*0.65) * bar_width
            positions = x_positions + offset

            # Get color for the current algorithm and feature selection
            color = color_palette[algo][method_idx]

            # Plot R^2 bars
            ax.bar(positions, subset['r2'], width=bar_width, color=color, label=f'{algo}-{method} R^2')

    # Configure R^2 plot
    ax.set_xticks(x_positions + (len(algorithms) * len(feature_selections) * bar_width) / 2)
    ax.set_xticklabels(tissues_sorted_r2)
    ax.set_ylabel(r'$R^2$')
    ax.set_title(r'$R^2$ Grouped by Tissue, Algorithm, and Feature Selection')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"gtex_outputs/method_comparison_r2_{split_id}.png")
    plt.close(fig)

# Call the function
plot_grouped_bar_chart(compiled_df, split_id)
