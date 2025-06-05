import numpy as np
import pandas as pd

def read_file(path, mode= 0):
     if mode == 0:
         dataset = pd.read_csv(path)

     if mode == 1:
         dataset = pd.concat(map(pd.read_csv, path), ignore_index=True)
             
     return dataset

def dataset_split(dataset, split):
    # Conver to decimals
    train_percent = split[0] / 100
    test_percent = split[1] / 100
    val_percent = split[2] / 100

    # Calculate number of samples for each split
    total_samples = len(dataset)
    train_samples = int(total_samples * train_percent)
    test_samples = int(total_samples * test_percent)
    val_samples = int(train_samples * val_percent)

    # Randomly shuffle indices
    indices = np.random.permutation(total_samples)
    train_initial_indices = indices[:train_samples]
    test_indices = indices[train_samples:train_samples + test_samples]

    # Create datasets using the indices
    train_initial_data = dataset.iloc[train_initial_indices]
    test_data = dataset.iloc[test_indices]

    train_indices = indices[:train_samples - val_samples]
    val_indices = indices[:val_samples]

    train_data = dataset.iloc[train_indices]
    val_data = dataset.iloc[val_indices]
    
    return train_data, test_data, val_data