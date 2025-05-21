import os
import datasets

#dataset_path and dataset_name maps to the args path and name in load_dataset

#def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
#    return dataset.shuffle(int(os.environ["DATA_SEED"]))

import os
import pandas as pd
from datasets import Dataset

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.shuffle(int(os.environ["DATA_SEED"]))



# def process_docs(dataset: Dataset) -> Dataset:
#     # Step 1: Load the CSV file containing the "ppl" column
#     csv_file_path = "/home/gsu/workspace/mergebase/data_script/processed_gsm_plus_with_generated_mathmodel.csv"  #os.environ["CSV_FILE_PATH"]  # Ensure this is set correctly in the environment
#     csv_data = pd.read_csv(csv_file_path)
    
#     # Step 2: Merge the dataset with the "ppl" column based on a common key
#     # Assuming both the dataset and CSV have a common column like "id"
#     df_dataset = dataset.to_pandas()  # Convert the dataset to a pandas DataFrame
#     merged_data = pd.merge(df_dataset, csv_data[['question', 'perplexity']], on='question', how='inner')
    
#     # Convert the merged DataFrame back to a Dataset
#     aligned_dataset = Dataset.from_pandas(merged_data)
    
#     # Step 3: Shuffle the dataset using the seed from the environment variable
#     shuffled_dataset = aligned_dataset.shuffle(seed=int(os.environ["DATA_SEED"]))
    
#     # Step 4: Select the first 1000 samples
#     sampled_dataset = shuffled_dataset.select(range(500))
    
#     # Step 5: Sort the sampled dataset by the "ppl" column
#     sorted_dataset = sampled_dataset.sort("perplexity")
    
#     # Step 6: Return the processed dataset
#     return sorted_dataset

