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


