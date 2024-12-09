# utils/data_processing.py

import os
import numpy as np
import pandas as pd


def load_data(file_path):
    """
    Load time-series data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        np.ndarray: Loaded data as a NumPy array.
    """
    df = pd.read_csv(file_path)
    data = df.values
    # Assuming the CSV has columns: timestamp, price
    # Reshape or preprocess as needed
    # For example, if multiple features:
    # data = df[['price', 'volume', ...]].values
    return data


def save_generated_data(generated_data, output_path):
    """
    Save generated data to a CSV file.

    Args:
        generated_data (list or np.ndarray): Generated time-series data.
        output_path (str): Path to save the CSV file.
    """
    # Flatten the list of arrays into a single DataFrame
    # This implementation assumes each generated sample is a 2D array (seq_len, features)
    flattened_data = []
    for sample in generated_data:
        for row in sample:
            flattened_data.append(row)
    df = pd.DataFrame(flattened_data, columns=[f'feature_{i}' for i in range(flattened_data[0].shape[0])])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """
    Divide original and synthetic data into training and testing sets.

    Args:
        data_x (list): Original data.
        data_x_hat (list): Generated data.
        data_t (list): Original time information.
        data_t_hat (list): Generated time information.
        train_rate (float, optional): Ratio of training data. Defaults to 0.8.

    Returns:
        tuple: Train and test splits for original and synthetic data.
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # Divide train/test index (synthetic data)
    no_hat = len(data_x_hat)
    idx_hat = np.random.permutation(no_hat)
    train_idx_hat = idx_hat[: int(no_hat * train_rate)]
    test_idx_hat = idx_hat[int(no_hat * train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx_hat]
    test_x_hat = [data_x_hat[i] for i in test_idx_hat]
    train_t_hat = [data_t_hat[i] for i in train_idx_hat]
    test_t_hat = [data_t_hat[i] for i in test_idx_hat]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat
