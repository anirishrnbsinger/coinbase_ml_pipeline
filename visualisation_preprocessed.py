

# visualisation_preprocessed.py

import streamlit as st
import pandas as pd
import psycopg2
from psycopg2 import pool
import plotly.graph_objects as go
import plotly.express as px

# TimescaleDB configuration
DB_NAME = 'crypto_data'
DB_USER = 'rishi'
DB_PASSWORD = '@password123'
DB_HOST = 'localhost'
DB_PORT = '5432'

# Initialize a connection pool
connection_pool = pool.SimpleConnectionPool(1, 20,
                                            dbname=DB_NAME,
                                            user=DB_USER,
                                            password=DB_PASSWORD,
                                            host=DB_HOST,
                                            port=DB_PORT)

# Columns
ticker_batch_columns = ['time', 'price', 'volume_24_h', 'low_24_h', 'high_24_h', 'low_52_w', 'high_52_w', 'price_percent_chg_24_h']
candle_columns = ['start', 'open', 'high', 'low', 'close', 'volume']

# Function to get data from the database
def get_preprocessed_data(product_id):
    sanitized_product_id = product_id.replace('-', '_')
    table_name = f"{sanitized_product_id}_preprocessed"
    
    conn = connection_pool.getconn()
    try:
        cur = conn.cursor()
        query = f"SELECT * FROM {table_name} ORDER BY time DESC LIMIT 1000"
        cur.execute(query)
        data = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
    except Exception as e:
        st.error(f"Error retrieving data for {product_id}: {str(e)}")
        data = pd.DataFrame()
    finally:
        cur.close()
        connection_pool.putconn(conn)
    
    return data

# Function to get ticker_batch data from the database
def get_ticker_batch_data(product_id):
    sanitized_product_id = product_id.replace('-', '_')
    table_name = f"ticker_batch_{sanitized_product_id}"
    
    conn = connection_pool.getconn()
    try:
        cur = conn.cursor()
        query = f"SELECT time, price, volume_24_h, low_24_h, high_24_h, low_52_w, high_52_w, price_percent_chg_24_h FROM {table_name} ORDER BY time DESC LIMIT 1000"
        cur.execute(query)
        data = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
    except Exception as e:
        st.error(f"Error retrieving ticker batch data for {product_id}: {str(e)}")
        data = pd.DataFrame()
    finally:
        cur.close()
        connection_pool.putconn(conn)
    
    return data

# Function to get candle data from the database
def get_candle_data(product_id):
    sanitized_product_id = product_id.replace('-', '_')
    table_name = f"candles_{sanitized_product_id}"
    
    conn = connection_pool.getconn()
    try:
        cur = conn.cursor()
        query = f"SELECT * FROM {table_name} ORDER BY start DESC LIMIT 1000"
        cur.execute(query)
        data = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
    except Exception as e:
        st.error(f"Error retrieving candle data for {product_id}: {str(e)}")
        data = pd.DataFrame()
    finally:
        cur.close()
        connection_pool.putconn(conn)
    
    return data

# Function to create continuous time range and merge data
def create_continuous_time_range_and_merge(data, time_column):
    if data.empty:
        return data

    # Round timestamps to the nearest second
    data[time_column] = pd.to_datetime(data[time_column]).dt.round('s')

    min_time = data[time_column].min()
    max_time = data[time_column].max()

    # Create a continuous range of timestamps with second-by-second frequency
    continuous_time_range = pd.date_range(start=min_time, end=max_time, freq='s')  # 'S' for second frequency

    # Create a DataFrame with continuous timestamps
    continuous_df = pd.DataFrame({time_column: continuous_time_range})

    # Merge the continuous timestamps with the actual data
    merged_data = pd.merge(continuous_df, data, on=time_column, how='left')

    return merged_data

def split_dataframe(df, chunk_size):
    chunks = []
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

# Streamlit app
st.title("Crypto Data Visualisation")

product_ids = ["BTC-USD", "ETH-USD", "MATIC-USD", "SOL-USD", "DAI-USD", "DOGE-USD",
               "ETH-BTC", "MATIC-BTC", "SOL-BTC", "DOGE-BTC", "SOL-ETH", "ETH-DAI"]

# Initialize session state for view_preprocessed if not already set
if 'view_preprocessed' not in st.session_state:
    st.session_state['view_preprocessed'] = True
if 'all_from_source' not in st.session_state:
    st.session_state['all_from_source'] = False
if 'selected_pairs' not in st.session_state:
    st.session_state['selected_pairs'] = product_ids  # Default to all available pairs

# Button to toggle between preprocessed and raw data
if st.button("Preprocessed", key="toggle_button"):
    st.session_state['view_preprocessed'] = not st.session_state['view_preprocessed']
    st.session_state['all_from_source'] = False  # Reset "All from Source" state

if not st.session_state['view_preprocessed']:
    st.session_state['all_from_source'] = st.checkbox("All from Source")

    if st.session_state['all_from_source']:
        st.session_state['selected_pairs'] = st.multiselect(
            "Select Multiple Pairs",
            product_ids,
            default=product_ids  # Default to all available pairs
        )

        if st.session_state['selected_pairs']:
            selected_product_id = None  # Disable single pair dropdown if multiple pairs are selected
        else:
            selected_product_id = st.selectbox("Select Product ID", product_ids)
    else:
        selected_product_id = st.selectbox("Select Product ID", product_ids)
else:
    selected_product_id = st.selectbox("Select Product ID", product_ids)

if not st.session_state['view_preprocessed']:
    use_ticker_batch = st.checkbox("Ticker Batch", value=True)
    use_candles = st.checkbox("Candles", value=True)

    data = pd.DataFrame()

    if st.session_state['selected_pairs']:
        for pair in st.session_state['selected_pairs']:
            if use_ticker_batch:
                ticker_data = get_ticker_batch_data(pair)
                data = pd.concat([data, ticker_data], axis=0)
            if use_candles:
                candle_data = get_candle_data(pair)
                data = pd.concat([data, candle_data], axis=0)
    else:
        if selected_product_id:
            if use_ticker_batch:
                ticker_data = get_ticker_batch_data(selected_product_id)
                data = pd.concat([data, ticker_data], axis=0)
            if use_candles:
                candle_data = get_candle_data(selected_product_id)
                data = pd.concat([data, candle_data], axis=0)
else:
    data = get_preprocessed_data(selected_product_id)

if not data.empty:
    if st.session_state['view_preprocessed']:
        available_columns = data.columns.tolist()
    else:
        # THIS AREA IS TO DETERMINE WHERE WE ARE MISSING CANDLES OR TICKER_BATCH DATA
        if st.session_state['all_from_source']:
            pairs_data = {}
            if use_candles:
                candles_data = {"candle_" + pair: get_candle_data(pair) for pair in st.session_state['selected_pairs']}
                pairs_data.update(candles_data)
            if use_ticker_batch:
                ticker_batch_data = {"ticker_batch_" + pair: get_ticker_batch_data(pair) for pair in st.session_state['selected_pairs']}
                pairs_data.update(ticker_batch_data)
        
            fig = go.Figure()
        
            for idx, (pair, pair_data) in enumerate(sorted(pairs_data.items())):
                pair_data = create_continuous_time_range_and_merge(pair_data, 'time' if 'time' in pair_data.columns else 'start')
                pair_data['idx'] = idx  # Add a constant y-value for each pair
                if 'time' in pair_data.columns:
                    fig.add_trace(go.Scatter(
                        x=pair_data['time'],
                        y=pair_data['idx'],
                        mode='lines',
                        name=pair,
                        connectgaps=False,
                        line=dict(color=px.colors.qualitative.Plotly[idx % len(px.colors.qualitative.Plotly)])
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=pair_data['start'],
                        y=pair_data['idx'],
                        mode='lines',
                        name=pair,
                        connectgaps=False,
                        line=dict(color=px.colors.qualitative.Plotly[idx % len(px.colors.qualitative.Plotly)])
                    ))
        
            fig.update_layout(
                title="Data Availability for Selected Pairs",
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(sorted(pairs_data.keys())))),
                    ticktext=list(sorted(pairs_data.keys())),
                    title="Pairs"
                ),
                xaxis_title="Time"
            )
        
            st.plotly_chart(fig, use_container_width=True)
            available_columns = []  # To avoid multiselect in this mode
        else:
            if use_ticker_batch and use_candles:
                available_columns = list(set(ticker_batch_columns + candle_columns))
            elif use_ticker_batch:
                available_columns = ticker_batch_columns
            elif use_candles:
                available_columns = candle_columns
            else:
                available_columns = []

    if available_columns:
        columns_to_display = st.multiselect("Select Columns to Display", available_columns, default=available_columns)

        if columns_to_display:
            columns_to_display = [col for col in columns_to_display if col in available_columns]
            data = create_continuous_time_range_and_merge(data, 'time' if 'time' in data.columns else 'start')
            chunks = split_dataframe(data, chunk_size=100000)  # Adjust chunk_size as needed

            fig = go.Figure()

            for chunk in chunks:
                if 'time' in chunk.columns:
                    fig.add_trace(go.Scatter(
                        x=chunk['time'],
                        y=chunk[columns_to_display[0]],
                        mode='lines',
                        name=selected_product_id,
                        connectgaps=False,
                        line=dict(color='blue')
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=chunk['start'],
                        y=chunk[columns_to_display[0]],
                        mode='lines',
                        name=selected_product_id,
                        connectgaps=False,
                        line=dict(color='blue')
                    ))

            fig.update_layout(
                title=f"{selected_product_id} Data",
                xaxis=dict(title='Time'),
                yaxis=dict(title='Value')
            )

            fig.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data available for the selected product.")