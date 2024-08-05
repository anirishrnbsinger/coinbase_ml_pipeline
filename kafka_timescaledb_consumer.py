
# kafka_timescaledb_consumer.py

import json
from confluent_kafka import Consumer, KafkaError
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd

# Kafka configuration
KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPICS = ['crypto_ticker_batch', 'crypto_candles', 'crypto_heartbeats']

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

# Create the heartbeats table if it does not exist
heartbeat_table_creation_query = """
CREATE TABLE IF NOT EXISTS heartbeats (
    timestamp TIMESTAMPTZ PRIMARY KEY,
    heartbeat_id TEXT
);
"""
try:
    cur.execute(heartbeat_table_creation_query)
    conn.commit()
    print("Heartbeat table created successfully")
except Exception as e:
    print(f"Error creating heartbeat table: {e}")
    conn.rollback()

def sanitize_product_id(product_id):
    return product_id.replace('-', '_')

def table_exists(table_name):
    cur.execute(f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = %s
        );
    """, (table_name,))
    return cur.fetchone()[0]

def create_table_if_not_exists(table_type, product_id):
    sanitized_product_id = sanitize_product_id(product_id)
    table_name = f"{table_type}_{sanitized_product_id}"
    if table_exists(table_name):
        return
    if table_type == "ticker_batch":
        table_creation_query = f"""
        CREATE TABLE {table_name} (
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
    elif table_type == "candles":
        table_creation_query = f"""
        CREATE TABLE {table_name} (
            start TIMESTAMPTZ PRIMARY KEY,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume DOUBLE PRECISION
        );
        """
    try:
        cur.execute(table_creation_query)
        conn.commit()
        print(f"Table {table_name} created successfully")
    except Exception as e:
        #print(f"Error creating table {table_name}: {e}")
        conn.rollback()

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

def store_candle_data(data):
    product_id = data['product_id']
    sanitized_product_id = sanitize_product_id(product_id)
    create_table_if_not_exists("candles", sanitized_product_id)
    table_name = f"candles_{sanitized_product_id}"
    try:
        records = [(data['start'], data['open'], data['high'], data['low'], data['close'], data['volume'])]
        execute_values(cur, f"""
            INSERT INTO {table_name} (start, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT (start) DO NOTHING
        """, records)
        conn.commit()
        for record in records:
            print(f"C: {product_id}, {pd.to_datetime(record[0])}, Open: {record[1]}, H: {record[2]}, L: {record[3]}, Close: {record[4]}, Vol: {record[5]}")
    except Exception as e:
        print(f"Error in store_candle_data for {product_id}: {e}")
        conn.rollback()

def store_heartbeat_data(data):
    try:
        records = [(data['timestamp'], data['heartbeat_id'])]
        execute_values(cur, """
            INSERT INTO heartbeats (timestamp, heartbeat_id)
            VALUES %s
            ON CONFLICT (timestamp) DO NOTHING
        """, records)
        conn.commit()
        print(f"Inserted heartbeat data: {records}")
    except Exception as e:
        print(f"Error in store_heartbeat_data: {e}")
        conn.rollback()

def main():
    consumer_conf = {
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': 'my_consumer_group',
        'auto.offset.reset': 'earliest'
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe(KAFKA_TOPICS)

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(msg.error())
                    break
            data = json.loads(msg.value().decode('utf-8'))
            topic = msg.topic()
            if topic == 'crypto_ticker_batch':
                store_ticker_batch_data(data)
            elif topic == 'crypto_candles':
                store_candle_data(data)
            elif topic == 'crypto_heartbeats':
                store_heartbeat_data(data)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()

if __name__ == "__main__":
    main()