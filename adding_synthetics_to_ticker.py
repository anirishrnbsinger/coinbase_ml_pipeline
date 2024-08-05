
# adding_synthetics_to_ticker.py

import psycopg2
import pandas as pd
from psycopg2.extras import execute_values
import numpy as np
from pykalman import KalmanFilter

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

def sanitize_product_id(product_id):
    return product_id.replace('-', '_')

def fetch_clean_ticker_data(sanitized_product_id):
    table_name = f"{sanitized_product_id}_clean_ticker"
    query = f"""
        SELECT * FROM {table_name}
        ORDER BY time
    """
    print(f"Fetching clean data for {sanitized_product_id} from {table_name}")
    try:
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        data = pd.DataFrame(rows, columns=columns)
        print(f"Fetched {len(data)} rows for {sanitized_product_id}")
        print(f"Columns in DataFrame: {data.columns.tolist()}")  # Add this line to check column names
        
        # Check for NaN values in the fetched data
        if data.isna().any().any():
            print(f"Initial data for {sanitized_product_id} contains NaN values:")
            print(data.isna().sum())
        
        return data
    except Exception as e:
        print(f"Error fetching data for {sanitized_product_id}: {e}")
        return pd.DataFrame()

def create_ticker_with_synthetics_table(sanitized_product_id):
    table_name = f"{sanitized_product_id}_tb_w_synthetics"
    table_creation_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            time TIMESTAMPTZ PRIMARY KEY,
            price DOUBLE PRECISION,
            volume_24_h DOUBLE PRECISION,
            low_24_h DOUBLE PRECISION,
            high_24_h DOUBLE PRECISION,
            low_52_w DOUBLE PRECISION,
            high_52_w DOUBLE PRECISION,
            price_percent_chg_24_h DOUBLE PRECISION,
            MA_5 DOUBLE PRECISION,
            MA_25 DOUBLE PRECISION,
            MA_50 DOUBLE PRECISION,
            MA_250 DOUBLE PRECISION,
            MA_500 DOUBLE PRECISION,
            MA_2500 DOUBLE PRECISION,
            MA_5000 DOUBLE PRECISION,
            EMA_5 DOUBLE PRECISION,
            EMA_25 DOUBLE PRECISION,
            EMA_50 DOUBLE PRECISION,
            EMA_250 DOUBLE PRECISION,
            EMA_500 DOUBLE PRECISION,
            EMA_2500 DOUBLE PRECISION,
            EMA_5000 DOUBLE PRECISION,
            RSI_14 DOUBLE PRECISION,
            RSI_140 DOUBLE PRECISION,
            RSI_1400 DOUBLE PRECISION,
            upper_band_20 DOUBLE PRECISION,
            lower_band_20 DOUBLE PRECISION,
            upper_band_200 DOUBLE PRECISION,
            lower_band_200 DOUBLE PRECISION,
            upper_band_2000 DOUBLE PRECISION,
            lower_band_2000 DOUBLE PRECISION,
            MACD_5_25 DOUBLE PRECISION,
            Signal_5_25 DOUBLE PRECISION,
            MACD_50_250 DOUBLE PRECISION,
            Signal_50_250 DOUBLE PRECISION,
            MACD_500_2500 DOUBLE PRECISION,
            Signal_500_2500 DOUBLE PRECISION,
            ROC_12 DOUBLE PRECISION,
            ROC_120 DOUBLE PRECISION,
            ROC_1200 DOUBLE PRECISION,
            MATR DOUBLE PRECISION,
            OBV DOUBLE PRECISION
        );
    """
    try:
        cur.execute(table_creation_query)
        conn.commit()
        print(f"Table {table_name} created successfully")
    except Exception as e:
        print(f"Error creating table {table_name}: {e}")
        conn.rollback()

def convert_data_types(df):
    print("convert_data_types: Converting DataFrame data types")
    for column in df.columns:
        if df[column].dtype == 'int64' or df[column].dtype == 'Int64':
            df[column] = df[column].astype(int)
        elif df[column].dtype == 'float64':
            df[column] = df[column].astype(float)
    print("convert_data_types: Conversion completed")
    return df

def store_ticker_with_synthetics_data(df, sanitized_product_id):
    table_name = f"{sanitized_product_id}_tb_w_synthetics"
    try:
        df = convert_data_types(df)  # Ensure correct data types
        records = df.to_records(index=False).tolist()  # Convert to list of tuples
        records = [[
            record[0].to_pydatetime() if hasattr(record[0], 'to_pydatetime') else record[0],  # Convert datetime if needed
            *[int(x) if isinstance(x, (np.int64, np.int32)) else float(x) if isinstance(x, (np.float64, np.float32)) else x for x in record[1:]]
        ] for record in records]  # Convert numpy types to native Python types

        execute_values(cur, f"""
            INSERT INTO {table_name} (time, price, volume_24_h, low_24_h, high_24_h, low_52_w, high_52_w, price_percent_chg_24_h,
                                      MA_5, MA_25, MA_50, MA_250, MA_500, MA_2500, MA_5000,
                                      EMA_5, EMA_25, EMA_50, EMA_250, EMA_500, EMA_2500, EMA_5000,
                                      RSI_14, RSI_140, RSI_1400,
                                      upper_band_20, lower_band_20, upper_band_200, lower_band_200, upper_band_2000, lower_band_2000,
                                      MACD_5_25, Signal_5_25, MACD_50_250, Signal_50_250, MACD_500_2500, Signal_500_2500,
                                      ROC_12, ROC_120, ROC_1200, MATR, OBV)
            VALUES %s
            ON CONFLICT (time) DO UPDATE SET
                price = EXCLUDED.price,
                volume_24_h = EXCLUDED.volume_24_h,
                low_24_h = EXCLUDED.low_24_h,
                high_24_h = EXCLUDED.high_24_h,
                low_52_w = EXCLUDED.low_52_w,
                high_52_w = EXCLUDED.high_52_w,
                price_percent_chg_24_h = EXCLUDED.price_percent_chg_24_h,
                MA_5 = EXCLUDED.MA_5,
                MA_25 = EXCLUDED.MA_25,
                MA_50 = EXCLUDED.MA_50,
                MA_250 = EXCLUDED.MA_250,
                MA_500 = EXCLUDED.MA_500,
                MA_2500 = EXCLUDED.MA_2500,
                MA_5000 = EXCLUDED.MA_5000,
                EMA_5 = EXCLUDED.EMA_5,
                EMA_25 = EXCLUDED.EMA_25,
                EMA_50 = EXCLUDED.EMA_50,
                EMA_250 = EXCLUDED.EMA_250,
                EMA_500 = EXCLUDED.EMA_500,
                EMA_2500 = EXCLUDED.EMA_2500,
                EMA_5000 = EXCLUDED.EMA_5000,
                RSI_14 = EXCLUDED.RSI_14,
                RSI_140 = EXCLUDED.RSI_140,
                RSI_1400 = EXCLUDED.RSI_1400,
                upper_band_20 = EXCLUDED.upper_band_20,
                lower_band_20 = EXCLUDED.lower_band_20,
                upper_band_200 = EXCLUDED.upper_band_200,
                lower_band_200 = EXCLUDED.lower_band_200,
                upper_band_2000 = EXCLUDED.upper_band_2000,
                lower_band_2000 = EXCLUDED.lower_band_2000,
                MACD_5_25 = EXCLUDED.MACD_5_25,
                Signal_5_25 = EXCLUDED.Signal_5_25,
                MACD_50_250 = EXCLUDED.MACD_50_250,
                Signal_50_250 = EXCLUDED.Signal_50_250,
                MACD_500_2500 = EXCLUDED.MACD_500_2500,
                Signal_500_2500 = EXCLUDED.Signal_500_2500,
                ROC_12 = EXCLUDED.ROC_12,
                ROC_120 = EXCLUDED.ROC_120,
                ROC_1200 = EXCLUDED.ROC_1200,
                MATR = EXCLUDED.MATR,
                OBV = EXCLUDED.OBV
        """, records)
        conn.commit()
        print(f"Data with synthetics stored in table {table_name}")
    except Exception as e:
        print(f"Error storing data with synthetics for {sanitized_product_id}: {e}")
        conn.rollback()



# Synthetic feature calculation functions with NaN handling using Kalman Filter
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

def exponential_moving_average(data_array, periods):
    print("exponential_moving_average: Starting calculation")
    df = pd.DataFrame({'data': data_array})
    ema_data = {}
    for period in periods:
        print(f"exponential_moving_average: Calculating for period {period}")
        ema_series = df['data'].ewm(span=period, adjust=False).mean()
        if ema_series.isna().any():
            print(f"exponential_moving_average: NaN values found for period {period}")
            
            # Initialize Kalman filter
            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
            
            # Fit the filter to the data
            observations = ema_series.bfill().values.reshape(-1, 1)  # Filling NaNs with 0 for initial fit
            kf = kf.em(observations, n_iter=5)
            
            # Smooth the data
            smoothed_state_means, _ = kf.smooth(observations)
            
            # Replace NaNs in ema_series with the smoothed values
            ema_series = pd.Series(smoothed_state_means.flatten(), index=ema_series.index)
            print(f"exponential_moving_average: NaN values have been smoothed using Kalman filter for period {period}")
        ema_data[f'EMA_{period}'] = ema_series
    print("exponential_moving_average: Calculation completed")
    return ema_data

def relative_strength_index(data_array, periods):
    print("relative_strength_index: Starting calculation")
    df = pd.DataFrame({'data': data_array})
    rsi_data = {}
    for period in periods:
        print(f"relative_strength_index: Calculating for period {period}")
        delta = df['data'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        if rsi_series.isna().any():
            print(f"relative_strength_index: NaN values found for period {period}")
            
            # Initialize Kalman filter
            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
            
            # Fit the filter to the data
            observations = rsi_series.fillna(0).values.reshape(-1, 1)  # Filling NaNs with 0 for initial fit
            kf = kf.em(observations, n_iter=5)
            
            # Smooth the data
            smoothed_state_means, _ = kf.smooth(observations)
            
            # Replace NaNs in rsi_series with the smoothed values
            rsi_series = pd.Series(smoothed_state_means.flatten(), index=rsi_series.index)
            print(f"relative_strength_index: NaN values have been smoothed using Kalman filter for period {period}")
        rsi_data[f'RSI_{period}'] = rsi_series
    print("relative_strength_index: Calculation completed")
    return rsi_data

def compute_bollinger_bands(df, column, periods):
    print("compute_bollinger_bands: Starting calculation")
    bollinger_data = {}
    for period in periods:
        print(f"compute_bollinger_bands: Calculating for period {period}")

        # Calculate rolling mean and standard deviation
        rolling_mean = df[column].rolling(window=period).mean()
        rolling_std = df[column].rolling(window=period).std()

        # Calculate upper and lower bands
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)

        if upper_band.isna().any() or lower_band.isna().any():
            print(f"compute_bollinger_bands: NaN values found for period {period}")

            # Forward fill then backfill NaN values to handle initial NaNs
            observations_upper = upper_band.ffill().bfill().values.reshape(-1, 1)
            observations_lower = lower_band.ffill().bfill().values.reshape(-1, 1)

            # Ensure no NaNs are present after filling
            if np.isnan(observations_upper).any() or np.isnan(observations_lower).any():
                raise ValueError(f"compute_bollinger_bands: NaNs present in observations for period {period} even after filling.")

            # Initialize Kalman filter
            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

            # Fit the filter to the upper band data
            kf_upper = kf.em(observations_upper, n_iter=5)
            smoothed_upper_means, _ = kf_upper.smooth(observations_upper)

            # Fit the filter to the lower band data
            kf_lower = kf.em(observations_lower, n_iter=5)
            smoothed_lower_means, _ = kf_lower.smooth(observations_lower)

            # Replace NaNs in bands with the smoothed values
            upper_band = pd.Series(smoothed_upper_means.flatten(), index=upper_band.index)
            lower_band = pd.Series(smoothed_lower_means.flatten(), index=lower_band.index)
            print(f"compute_bollinger_bands: NaN values have been smoothed using Kalman filter for period {period}")

        bollinger_data[f'upper_band_{period}'] = upper_band
        bollinger_data[f'lower_band_{period}'] = lower_band

    print("compute_bollinger_bands: Calculation completed")
    return bollinger_data


def macd(data, fast_periods, slow_periods):
    print("macd: Starting calculation")
    macd_data = {}
    df = pd.DataFrame({'data': data})
    for fast_period, slow_period in zip(fast_periods, slow_periods):
        print(f"macd: Calculating for fast period {fast_period} and slow period {slow_period}")
        fast_ema = df['data'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df['data'].ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        if macd_line.isna().any() or signal_line.isna().any():
            print(f"macd: NaN values found for fast period {fast_period} and slow period {slow_period}")
            
            # Initialize Kalman filter
            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
            
            # Fit the filter to the data
            observations_macd = macd_line.bfill().values.reshape(-1, 1)  # Filling NaNs with 0 for initial fit
            observations_signal = signal_line.bfill().values.reshape(-1, 1)
            kf = kf.em(observations_macd, n_iter=5)
            
            # Smooth the data
            smoothed_macd_means, _ = kf.smooth(observations_macd)
            smoothed_signal_means, _ = kf.smooth(observations_signal)
            
            # Replace NaNs in lines with the smoothed values
            macd_line = pd.Series(smoothed_macd_means.flatten(), index=macd_line.index)
            signal_line = pd.Series(smoothed_signal_means.flatten(), index=signal_line.index)
            print(f"macd: NaN values have been smoothed using Kalman filter for fast period {fast_period} and slow period {slow_period}")
        
        macd_data[f'MACD_{fast_period}_{slow_period}'] = macd_line
        macd_data[f'Signal_{fast_period}_{slow_period}'] = signal_line
    print("macd: Calculation completed")
    return macd_data


def rate_of_change(data_array, periods):
    #print("rate_of_change: Starting calculation")
    df = pd.DataFrame({'data': data_array})
    roc_data = {}
    for period in periods:
        #print(f"rate_of_change: Calculating for period {period}")
        roc_series = df['data'].pct_change(periods=period) * 100
        if roc_series.isna().any():
            #print(f"rate_of_change: NaN values found for period {period} at indices: {roc_series[roc_series.isna()].index.tolist()}")
            #print(f"rate_of_change: Data segment causing NaNs: {df['data'].iloc[roc_series[roc_series.isna()].index.tolist()]}")
            
            # Initialize Kalman filter
            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
            
            # Fit the filter to the data
            observations = roc_series.bfill().values.reshape(-1, 1)  # Filling NaNs 
            kf = kf.em(observations, n_iter=5)
            
            # Smooth the data
            smoothed_state_means, _ = kf.smooth(observations)
            
            # Replace NaNs in roc_series with the smoothed values
            roc_series = pd.Series(smoothed_state_means.flatten(), index=roc_series.index)
            #print(f"rate_of_change: NaN values have been smoothed using Kalman filter for period {period}")
        if roc_series.isna().any():
            print(f"rate_of_change: NaN values still present after smoothing for period {period}")
        else:
            print(f"rate_of_change: No NaN values found for period {period}")
        roc_data[f'ROC_{period}'] = roc_series
    print("rate_of_change: Calculation completed")
    return roc_data

def modified_average_true_range(high_24_h, low_24_h):
    print("modified_average_true_range: Starting calculation")
    atr = high_24_h - low_24_h
    if np.isnan(atr).any():
        print("modified_average_true_range: NaN values found")
    print("modified_average_true_range: Calculation completed")
    return atr


def on_balance_volume(price, volume_24_h):
    print("on_balance_volume: Starting calculation")
    df = pd.DataFrame({'price': price, 'volume': volume_24_h})
    obv = (df['volume'].where(df['price'].diff() > 0, -df['volume'])).cumsum()
    if obv.isna().any():
        print("on_balance_volume: NaN values found")
    print("on_balance_volume: Calculation completed")
    return obv

def compute_synthetic_features(df, col_prefix, high_col, low_col, close_col, volume_col, periods):
    print("compute_synthetic_features: Starting calculation")
    # Define extended periods for RSI, Bollinger Bands, Rate of Change, and Average True Range
    extended_periods = periods + [period * 10 for period in periods] + [period * 100 for period in periods]

    ma_data = moving_average(df[close_col].values, periods)
    for period in periods:
        df[f'{col_prefix}_MA_{period}'] = ma_data[f'MA_{period}']

    ema_data = exponential_moving_average(df[close_col].values, periods)
    for period in periods:
        df[f'{col_prefix}_EMA_{period}'] = ema_data[f'EMA_{period}']

    rsi_data = relative_strength_index(df[close_col].values, [14, 140, 1400])
    for period in rsi_data:
        df[f'{col_prefix}_{period}'] = rsi_data[period]

    bollinger_data = compute_bollinger_bands(df, close_col, [20, 200, 2000])
    for key in bollinger_data:
        df[f'{col_prefix}_{key}'] = bollinger_data[key]

    macd_data = macd(df[close_col].values, [5, 50, 500], [25, 250, 2500])
    for key in macd_data:
        df[f'{col_prefix}_{key}'] = macd_data[key]

    roc_data = rate_of_change(df[close_col].values, [12, 120, 1200])
    for key in roc_data:
        df[f'{col_prefix}_{key}'] = roc_data[key]

    df['MATR'] = modified_average_true_range(df[high_col].values, df[low_col].values)

    print("compute_synthetic_features: Adding OBV")

    df['OBV'] = on_balance_volume(df[close_col].values, df[volume_col].values)

    print("compute_synthetic_features: Calculation completed")
    return df

def main():
    products = ["BTC-USD", "ETH-USD", "MATIC-USD", "SOL-USD", "DAI-USD", 
                "DOGE-USD", "ETH-BTC", "MATIC-BTC", "SOL-BTC", "DOGE-BTC", 
                "SOL-ETH", "ETH-DAI"]

    for product_id in products:
        sanitized_product_id = sanitize_product_id(product_id)
        data = fetch_clean_ticker_data(sanitized_product_id)
        if data.empty:
            print(f"No data available for {sanitized_product_id}")
            continue
        
        data_with_synthetics = compute_synthetic_features(data, '', 'high_24_h', 'low_24_h', 'price', 'volume_24_h', [5, 25, 50, 250, 500, 2500, 5000])
        create_ticker_with_synthetics_table(sanitized_product_id)
        store_ticker_with_synthetics_data(data_with_synthetics, sanitized_product_id)
        print(f"Processed and stored data with synthetics for {sanitized_product_id}")

if __name__ == "__main__":
    main()