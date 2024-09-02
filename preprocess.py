import pandas as pd

def separate_age_ranges(filepath, output_lower, output_upper):
    # Read the data
    df = pd.read_csv(filepath, sep='\s+')
    
    # Extract and separate the AGE column
    age_series = df['AGE']
    lower_values = age_series.apply(lambda x: x.split('-')[0]).astype(float)
    upper_values = age_series.apply(lambda x: x.split('-')[1]).astype(float)
    
    # Replace AGE column with lower values in one DataFrame
    df_lower = df.copy()
    df_lower['AGE'] = lower_values
    
    # Replace AGE column with upper values in another DataFrame
    df_upper = df.copy()
    df_upper['AGE'] = upper_values
    
    # Save the DataFrames to CSV files
    df_lower.to_csv(output_lower, index=False, sep='\t')
    df_upper.to_csv(output_upper, index=False, sep='\t')
    
    print(f"Lower age values saved to {output_lower}")
    print(f"Upper age values saved to {output_upper}")

# Example usage
separate_age_ranges(
    filepath="../../../gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt",
    output_lower="data_with_lower_age.csv",
    output_upper="data_with_upper_age.csv"
)
