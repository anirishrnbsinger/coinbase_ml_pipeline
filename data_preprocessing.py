
# data preprocessing.py

import psycopg2
import pandas as pd
from pykalman import KalmanFilter
from datetime import timedelta
from psycopg2.extras import execute_values

# TimescaleDB configuration
DB_NAME = 'crypto_data'
DB_USER = 'rishi'
DB_PASSWORD = '@password123'
DB_HOST = 'localhost'
DB_PORT = '5432'

# Create a connection to TimescaleDB
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()

TOLERANCE_SECONDS = 600

def sanitize_product_id(product_id):
    return product_id.replace('-', '_')

def fetch_ticker_batch_data(product_id, start_time, end_time):
    sanitized_product_id = sanitize_product_id(product_id)
    table_name = f"ticker_batch_{sanitized_product_id}"
    query = f"""
        SELECT * FROM {table_name}
        WHERE time BETWEEN %s AND %s
        ORDER BY time
    """
    print(f"Fetching data for {product_id} from {start_time} to {end_time}")
    try:
        cur.execute(query, (start_time, end_time))
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        data = pd.DataFrame(rows, columns=columns)
        print(f"Fetched {len(data)} rows for {product_id}")
        return data
    except Exception as e:
        print(f"Error fetching data for {product_id}: {e}")
        return pd.DataFrame()

def preprocess_data(df):
    print("Preprocessing data")
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').resample('s').mean().interpolate()
    print("Data preprocessing complete")
    return df

def identify_gaps(df):
    print("Identifying gaps in data")
    all_times = pd.date_range(start=df.index.min(), end=df.index.max(), freq='s')
    missing_times = all_times.difference(df.index)
    return missing_times

def find_valid_time_range(products, tolerance_seconds=TOLERANCE_SECONDS):
    print("Finding valid time range")
    end_time = pd.Timestamp.now()
    start_time = end_time - pd.Timedelta(days=7)  # Adjust as necessary

    while True:
        valid = True
        intervals = []

        for product_id in products:
            data = fetch_ticker_batch_data(product_id, start_time, end_time)
            if data.empty:
                print(f"No data found for {product_id} in the given range")
                valid = False
                break
            preprocessed_data = preprocess_data(data)
            gaps = identify_gaps(preprocessed_data)
            if not gaps.empty and (gaps[1:] - gaps[:-1]).max().seconds > tolerance_seconds:
                print(f"Gap larger than {tolerance_seconds} seconds found for {product_id}")
                valid = False
                break
            intervals.append((preprocessed_data.index.min(), preprocessed_data.index.max()))
        
        if valid:
            common_start = max(interval[0] for interval in intervals)
            common_end = min(interval[1] for interval in intervals)
            if common_end - common_start > timedelta(seconds=0):
                print(f"Valid time range found: {common_start} to {common_end}")
                return common_start, common_end
        
        end_time -= pd.Timedelta(minutes=1)
        start_time = end_time - pd.Timedelta(days=1)

def fill_missing_data(df):
    print("Filling missing data using Kalman Filter")
    kf = KalmanFilter(initial_state_mean=df.iloc[0], n_dim_obs=df.shape[1])
    state_means, _ = kf.smooth(df.values)
    df.loc[:, :] = state_means
    print("Missing data filled")
    return df

def create_clean_ticker_table(sanitized_product_id):
    table_name = f"{sanitized_product_id}_clean_ticker"
    table_creation_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            time TIMESTAMPTZ PRIMARY KEY,
            price DOUBLE PRECISION,
            volume_24_h DOUBLE PRECISION,
            low_24_h DOUBLE PRECISION,
            high_24_h DOUBLE PRECISION,
            low_52_w DOUBLE PRECISION,
            high_52_w DOUBLE PRECISION,
            price_percent_chg_24_h DOUBLE PRECISION
        );
    """
    try:
        cur.execute(table_creation_query)
        conn.commit()
        print(f"Table {table_name} created successfully")
    except Exception as e:
        print(f"Error creating table {table_name}: {e}")
        conn.rollback()

def store_clean_ticker_data(df, sanitized_product_id):
    table_name = f"{sanitized_product_id}_clean_ticker"
    try:
        records = df.to_records(index=True)
        execute_values(cur, f"""
            INSERT INTO {table_name} (time, price, volume_24_h, low_24_h, high_24_h, low_52_w, high_52_w, price_percent_chg_24_h)
            VALUES %s
            ON CONFLICT (time) DO UPDATE SET
                price = EXCLUDED.price,
                volume_24_h = EXCLUDED.volume_24_h,
                low_24_h = EXCLUDED.low_24_h,
                high_24_h = EXCLUDED.high_24_h,
                low_52_w = EXCLUDED.low_52_w,
                high_52_w = EXCLUDED.high_52_w,
                price_percent_chg_24_h = EXCLUDED.price_percent_chg_24_h
        """, records)
        conn.commit()
        print(f"Clean data stored in table {table_name}")
    except Exception as e:
        print(f"Error storing clean data for {sanitized_product_id}: {e}")
        conn.rollback()

def main():
    products = ["BTC-USD", "ETH-USD", "MATIC-USD", "SOL-USD", "DAI-USD", 
                "DOGE-USD", "ETH-BTC", "MATIC-BTC", "SOL-BTC", "DOGE-BTC", 
                "SOL-ETH", "ETH-DAI"]

    start_time, end_time = find_valid_time_range(products)
    print(f"Valid time range for all products: {start_time} to {end_time}")

    for product_id in products:
        data = fetch_ticker_batch_data(product_id, start_time, end_time)
        if data.empty:
            print(f"No data available for {product_id} in the valid time range")
            continue
        preprocessed_data = preprocess_data(data)
        filled_data = fill_missing_data(preprocessed_data)
        sanitized_product_id = sanitize_product_id(product_id)
        create_clean_ticker_table(sanitized_product_id)
        store_clean_ticker_data(filled_data, sanitized_product_id)
        print(f"Processed and stored clean data for {product_id}")

if __name__ == "__main__":
    main()