
# load_data.py

import os
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import pool
import tensorflow as tf


# Set up a connection pool
connection_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=12,  # Adjust this based on your requirements and database limits
    dbname='crypto_data',
    user='rishi',
    password='@password123',
    host='localhost',
    port='5432'
)

def find_most_recent_file():
    files = [f for f in os.listdir('.') if f.endswith('_trimmed.csv')]
    if not files:
        raise FileNotFoundError("No trimmed CSV files found in the current directory.")
    
    latest_file = max(files, key=lambda f: datetime.strptime(f.split('_')[1], '%Y-%m-%d-%H-%M-%S'))
    return latest_file

def fetch_synthetic_ticker_data(sanitized_product_id, conn):
    table_name = f"{sanitized_product_id}_tb_w_synthetics"
    query = f"""
        SELECT * FROM {table_name}
        ORDER BY time
    """
    print(f"Fetching synthetic data for {sanitized_product_id} from {table_name}")
    try:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        data = pd.DataFrame(rows, columns=columns)
        
        # Convert 'time' to datetime and set it as index
        data['time'] = pd.to_datetime(data['time'], utc=True)
        data.set_index('time', inplace=True)
        
        print(f"Fetched {len(data)} rows for {sanitized_product_id}")
        print(f"Data range: {data.index.min()} to {data.index.max()}")
        
        # Check for NaN values in the fetched data
        if data.isna().any().any():
            print(f"Initial data for {sanitized_product_id} contains NaN values:")
            print(data.isna().sum())
        
        cur.close()
        return data
    except Exception as e:
        print(f"Error fetching data for {sanitized_product_id}: {e}")
        return pd.DataFrame()

def load_and_merge_data(product_ids, use_recent=False):
    print(f"load_and_merge_data: Loading and merging data for {product_ids}")
    
    if use_recent:
        file_name = find_most_recent_file()
        print(f"Using most recent file: {file_name}")
        combined_data = pd.read_csv(file_name, index_col=0, parse_dates=True)
        print(f"Loaded data shape: {combined_data.shape}")
        return combined_data

    data_frames = []
    start_times = []
    end_times = []

    try:
        # Acquire a connection from the pool
        conn = connection_pool.getconn()
        
        for product_id in product_ids:
            sanitized_product_id = product_id.replace('-', '_').lower()
            data = fetch_synthetic_ticker_data(sanitized_product_id, conn)
            if data.empty:
                print(f"No data available for {sanitized_product_id}")
                raise ValueError(f"No data available for {sanitized_product_id}")

            # Check for gaps in timestamps
            expected_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='s')
            missing_timestamps = expected_range.difference(data.index)
            if not missing_timestamps.empty:
                raise ValueError(f"Gaps found in timestamps for {product_id}. Missing timestamps: {missing_timestamps}")

            start_time = data.index.min()
            end_time = data.index.max()
            print(f"{product_id} data range: {start_time} to {end_time}")
            
            start_times.append(start_time)
            end_times.append(end_time)
            data_frames.append(data)

    except Exception as e:
        print(f"Error occurred during data loading: {e}")
    finally:
        # Release the connection back to the pool
        if conn:
            connection_pool.putconn(conn)

    # Determine the latest start and earliest end
    latest_start = max(start_times)
    earliest_end = min(end_times)
    print(f"Common data range: {latest_start} to {earliest_end}")

    # Check if the file already exists
    file_name = f"{latest_start.strftime('%Y-%m-%d-%H-%M-%S')}_{earliest_end.strftime('%Y-%m-%d-%H-%M-%S')}_trimmed.csv"
    if os.path.exists(file_name):
        print(f"File {file_name} already exists. Loading data from file.")
        combined_data = pd.read_csv(file_name, index_col=0, parse_dates=True)
        print(f"Loaded data shape: {combined_data.shape}")
        return combined_data

    # If the file doesn't exist, proceed with data processing
    print("File not found. Processing data...")

    # Verify that we have a valid date range
    if latest_start >= earliest_end:
        raise ValueError(f"No common date range found. Latest start ({latest_start}) is after earliest end ({earliest_end})")

    # Trim each dataframe to the common range
    trimmed_frames = []
    for df, product_id in zip(data_frames, product_ids):
        trimmed_df = df.loc[latest_start:earliest_end]
        trimmed_frames.append(trimmed_df)
        print(f"Trimmed data for {product_id}. New shape: {trimmed_df.shape}")

    # Combine the trimmed dataframes
    combined_data = pd.concat(trimmed_frames, axis=1, join='inner')

    # Verify that there are no NaN values in the combined data
    if combined_data.isna().any().any():
        nan_columns = combined_data.columns[combined_data.isna().any()].tolist()
        raise ValueError(f"NaN values found in columns after trimming: {nan_columns}")

    # Verify that all dataframes have the same length
    if not all(len(df) == len(combined_data) for df in trimmed_frames):
        raise ValueError("Not all trimmed dataframes have the same length")

    # Save combined data to a CSV file
    file_name = f"{latest_start.strftime('%Y-%m-%d-%H-%M-%S')}_{earliest_end.strftime('%Y-%m-%d-%H-%M-%S')}_trimmed.csv"
    combined_data.to_csv(file_name)
    print(f"Trimmed and combined data saved to {file_name}")
    print(f"Final combined data shape: {combined_data.shape}")

    return combined_data

def label_data_split(df, product_ids, horizons):
    num_pairs = len(product_ids)
    features_per_pair = 41
    #print(f"prepare_and_label_data: Initial dataframe shape: {df.shape}")
    largest_horizon = max(horizons)

    def generator():
        n = len(df)
        # need largest distance to make all 4 pred at once
        q_of_labelled = n - largest_horizon
        #print(f"prepare_and_label_data: Number of labeled points: {q_of_labelled}")
        
        for i in range(1000, q_of_labelled):
            # Input data
            start_index = i - 1000 # we introduce a limitation on the size of the X to save 
            X = df.iloc[start_index:i].values
            X = tf.convert_to_tensor(np.array(X), dtype=tf.float32)
            #print("X shape is", X.shape)
            #X = X.reshape(1, 1000, 492)  # 492 features (12 pairs * 41 features)
            
            
            # Labels
            y = []
            for horizon in horizons:
                label_index = i + horizon - 1
                y.append([df.iloc[label_index, j * features_per_pair] for j in range(num_pairs)])
            y = tf.convert_to_tensor(np.array(y).reshape(1, 48), dtype=tf.float32)
            #print("X is", X)
            #print("y is", y)
            #print("y shape is", y.shape)
            yield X, y
            #for j, pair in enumerate(product_ids):
            #    y = np.array(y)
            #    subset_of_y = y[:, j].reshape(1, 4)
            #    #print("labels for {pair} are", subset_of_y)
            #    yield pair, X, subset_of_y

    return generator