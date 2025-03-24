import os
import gzip
import csv

def transform_gct_to_csv(gct_gz_file, csv_file):
    with gzip.open(gct_gz_file, 'rt') as gct_file:
        lines = gct_file.readlines()

    header = lines[2].strip().split('\t')
    data_lines = lines[3:]

    with open(csv_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for line in data_lines:
            writer.writerow(line.strip().split('\t'))

def process_all_gct_files_in_directory(directory, output_directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.gct.gz'):
                gct_gz_file = os.path.join(root, file)
                
                relative_path = os.path.relpath(root, directory)
                output_dir = os.path.join(output_directory, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                
                csv_file = os.path.join(output_dir, file.replace('.gct.gz', '.csv'))
                
                transform_gct_to_csv(gct_gz_file, csv_file)
                print(f"Converted {gct_gz_file} to {csv_file}")

data_directory = 'TPM_data'
output_directory = 'TPM_dataCSV'
process_all_gct_files_in_directory(data_directory, output_directory)
