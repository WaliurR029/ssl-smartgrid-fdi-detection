# src/preprocessing.py

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_lcl_pivoted_2013(
    csv_path: str,
    num_households: int = 25,
    chunksize: int = 500_000
) -> pd.DataFrame:
    """
    Phase 2:
    Loads LCL in chunks, extracts 2013 records efficiently,
    and pivots to multivariate format (Time x Household Channels).
    """
    cols = ['LCLid', 'DateTime', 'KWH/hh (per half hour) ']
    records = []
    
    print(f"Streaming data from {csv_path} in chunks...")
    for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=chunksize, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        chunk = chunk[chunk['DateTime'].str.startswith('2013')]
        if not chunk.empty:
            records.append(chunk)
            
    print("Concatenating filtered 2013 chunks...")
    df_2013 = pd.concat(records, ignore_index=True)
    
    df_2013['KWH/hh (per half hour)'] = pd.to_numeric(df_2013['KWH/hh (per half hour)'], errors='coerce')
    df_2013['DateTime'] = pd.to_datetime(df_2013['DateTime'])
    df_2013 = df_2013.dropna()

    top_hh = df_2013['LCLid'].value_counts().head(num_households).index
    df_sub = df_2013[df_2013['LCLid'].isin(top_hh)]

    matrix = df_sub.pivot_table(
        index='DateTime',
        columns='LCLid',
        values='KWH/hh (per half hour)',
        aggfunc='mean'
    ).sort_index().ffill().bfill()

    return matrix


def split_and_scale_timeseries(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Phase 4:
    Chronological splitting and leakage-free MinMax scaling.
    MinMaxScaler is fitted ONLY on the training partition.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))

    # Fit scaler strictly on training data
    train_scaled = scaler.fit_transform(train_df.values)
    val_scaled = scaler.transform(val_df.values)
    test_scaled = scaler.transform(test_df.values)

    return train_scaled, val_scaled, test_scaled, scaler


def create_sliding_windows(
    data: np.ndarray,
    window_size: int = 48,
    step_size: int = 1
) -> np.ndarray:
    """
    Phase 4:
    Slices a 2D array (Time, Features) into 3D sliding windows:
    Shape: (samples, window_size, features)
    """
    num_samples = (len(data) - window_size) // step_size + 1
    windows = [
        data[i : i + window_size]
        for i in range(0, num_samples * step_size, step_size)
    ]
    return np.array(windows, dtype=np.float32)