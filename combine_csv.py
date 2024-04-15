import os
import pandas as pd
from tqdm import tqdm

data_by_group = {}


def reorganize_csv(input_file, output_dir, group_by_column):
    # Read the CSV file in chunks
    full_input_path = os.path.join(output_dir, input_file)
    reader = pd.read_csv(full_input_path, chunksize=100000)  # Adjust chunksize as needed

    # Iterate over the CSV chunks
    for chunk in tqdm(reader, desc="Processing chunks"):
        # Group the chunk by the specified column
        grouped = chunk.groupby(group_by_column)
        grouped_file = f"grouped_data{chunk}.csv"
        grouped_path = os.path.join(output_dir, grouped_file)
        grouped.to_csv(grouped_path)
        print(f"Done grouping for {chunk}")

        # Iterate over the groups and append data to the dictionary
        for group_value, group_data in grouped:
            if group_value not in data_by_group:
                data_by_group[group_value] = group_data
            else:
                data_by_group[group_value] = pd.concat([data_by_group[group_value], group_data], ignore_index=True)

    # Write the reorganized data to new CSV files
    for group_value, group_data in tqdm(data_by_group.items(), desc="Writing files"):
        output_file = f"{output_dir}/{group_value}.csv"
        group_data.to_csv(output_file, index=False)

    return data_by_group


def read_and_process_csv_files(folder_path, specific_files=None):
    for file in os.listdir(folder_path):
        if file.endswith('.csv') and (specific_files is None or file in specific_files):
            try:
                print(f"Start processing for {file}")
                reorganize_csv(file, folder_path, 'citingcorpusid')
                print('successfully reorganized')
            except pd.errors.ParserError as e:
                print(f"Error reading {file}: {e}")


folder_path = '../../share/garg/semantic_scholar_data/citations'
specific_files = ['citations_chunk_0.csv', 'citations_chunk_1.csv',
                  'citations_chunk_2.csv', 'citations_chunk_3.csv']
read_and_process_csv_files(folder_path, specific_files)
