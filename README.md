# coinbase_ml_pipeline
Taking data from the Coinbase API and experimenting with ML models.

We are taking data from Coinbase's websocket and passing it through Kafka to be stored and preprocessed:

```
def handle_ticker_batch_event(tickers, timestamp):
    try:
        for ticker in tickers:
            ticker['time'] = pd.to_datetime(timestamp).isoformat()
            print(f"handle_ticker_batch_event: {ticker}")  # Debug statement
            producer.produce('crypto_ticker_batch', json.dumps(ticker))
            producer.flush()
    except Exception as e:
        print(f"Error in handle_ticker_batch_event: {e}")
```
The script running the Kafka consumer then stores the data in a TimescaleDB running in PostgreSQL (locally) 

```
def store_ticker_batch_data(data):
    product_id = data['product_id']
    sanitized_product_id = sanitize_product_id(product_id)
    create_table_if_not_exists("ticker_batch", sanitized_product_id)
    table_name = f"ticker_batch_{sanitized_product_id}"
    try:
        records = [(data['time'], data['price'], data['volume_24_h'], data['low_24_h'], data['high_24_h'], data['low_52_w'], data['high_52_w'], data['price_percent_chg_24_h'])]
        execute_values(cur, f"""
            INSERT INTO {table_name} (time, price, volume_24_h, low_24_h, high_24_h, low_52_w, high_52_w, price_percent_chg_24_h)
            VALUES %s
            ON CONFLICT (time) DO NOTHING
        """, records)
        conn.commit()
        for record in records:
            print(f"TB: {product_id}, {record[0]}, Price: {record[1]}, Vol(24h): {record[2]}, L(24h): {record[3]}, H(24h): {record[4]}, L(52w): {record[5]}, H(52w): {record[6]}, %(24h): {record[7]}")
    except Exception as e:
        print(f"Error in store_ticker_batch_data for {product_id}: {e}")
        conn.rollback()
```

The ticker batch data being returned is then resampled down to Second frequency and gaps in the data are filled in with a Kalman Smoother:


```
def preprocess_data(df):
    print("Preprocessing data")
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').resample('s').mean().interpolate()
    print("Data preprocessing complete")
    return df

def fill_missing_data(df):
    print("Filling missing data using Kalman Filter")
    kf = KalmanFilter(initial_state_mean=df.iloc[0], n_dim_obs=df.shape[1])
    state_means, _ = kf.smooth(df.values)
    df.loc[:, :] = state_means
    print("Missing data filled")
    return df
```
Conventional metrics are synthesised from this data such as MA, EMA etc. (I will be continuing to adapt the use of these):
```
def moving_average(data_array, periods):
    print("moving_average: Starting calculation")
    df = pd.DataFrame({'data': data_array})
    ma_data = {}
    for period in periods:
        print(f"moving_average: Calculating for period {period}")
        ma_series = df['data'].rolling(window=period).mean()
        if ma_series.isna().any():
            print(f"moving_average: NaN values found for period {period}")
            
            # Initialize Kalman filter
            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
            
            # Fit the filter to the data
            observations = ma_series.bfill().values.reshape(-1, 1)  # Filling NaNs with 0 for initial fit
            kf = kf.em(observations, n_iter=5)
            
            # Smooth the data
            smoothed_state_means, _ = kf.smooth(observations)
            
            # Replace NaNs in ma_series with the smoothed values
            ma_series = pd.Series(smoothed_state_means.flatten(), index=ma_series.index)
            print(f"moving_average: NaN values have been smoothed using Kalman filter for period {period}")
        ma_data[f'MA_{period}'] = ma_series
    print("moving_average: Calculation completed")
    return ma_data

```

The data is then aligned to be preprocessed for training:

```
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
```

I have initially been feeding the data in sequentially up to 1000 consecutive seconds. The labels are then prices at 4 horizons in the future (10, 20, 100, 200) seconds. Due to the quantity of samples and future intentions to adapt this model and pipeline to Incremental Learning, I am feeding the data to the model by a generator:

```
def create_dataset(product_ids, use_recent):
    dataset = tf.data.Dataset.from_generator(
        lambda: load_data(product_ids, use_recent),
        output_signature=(
            tf.TensorSpec(shape=(1000, 492), dtype=tf.float32),
            tf.TensorSpec(shape=(1, 48), dtype=tf.float32)
        )
    )
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


def load_data(product_ids, use_recent = False):
    #logger.info(f"load_data: Loading and preparing data for {product_ids}")
    
    # First, align and merge data from all tables
    data = load_and_merge_data(product_ids, use_recent)
    #logger.info("load data: dimensions of combined data are", data.shape)
    
    # Load the aligned data from the saved CSV file
    horizons = [10, 20, 100, 200]
    data_generator = label_data_split(data, product_ids, horizons)
    
    for X_train, y in data_generator():

        scaler = tf.keras.layers.experimental.preprocessing.Normalization()
        scaler.adapt(X_train)
        
        yield X_train, y
```
