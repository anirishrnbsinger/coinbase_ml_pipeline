
# coinbase_kafka_producer.py

#region imports
import os
import time
import json
import jwt
import hashlib
import websocket
import certifi
from confluent_kafka import Producer
import pandas as pd
import logging
import threading

# Configure logging
logging.basicConfig(filename='websocket_connection.log', level=logging.INFO)

# Load sensitive variables from environment variables
API_KEY = os.getenv('COINBASE_API_KEY')
SIGNING_KEY = os.getenv('COINBASE_SIGNING_KEY')
ALGORITHM = "ES256"

if not SIGNING_KEY or not API_KEY:
    raise ValueError("Missing mandatory environment variable(s)")

CHANNEL_NAMES = {
    "candles": "candles",
    "ticker_batch": "ticker_batch",
    "heartbeats": "heartbeats",
    "user": "user"
}

WS_API_URL = "wss://advanced-trade-ws.coinbase.com"

# Initialize Kafka producer
producer_conf = {'bootstrap.servers': 'localhost:9092'}
producer = Producer(**producer_conf)

def generate_jwt():
    payload = {
        "iss": "coinbase-cloud",
        "nbf": int(time.time()),
        "exp": int(time.time()) + 120,
        "sub": API_KEY,
    }
    headers = {
        "kid": API_KEY,
        "nonce": hashlib.sha256(os.urandom(16)).hexdigest()
    }
    token = jwt.encode(payload, SIGNING_KEY, algorithm=ALGORITHM, headers=headers)
    return token

jwt_token = generate_jwt()


def on_error(ws, error):
    logging.error(f"Error in on_error: {error}")
    print(f"Error in on_error: {error}")

def on_close(ws, close_status_code, close_msg):
    disconnect_time = pd.Timestamp.now().isoformat()
    logging.info(f"WebSocket closed at {disconnect_time} with code: {close_status_code} and message: {close_msg}")
    print(f"WebSocket closed at {disconnect_time} with code: {close_status_code} and message: {close_msg}")
    logging.info("Reconnecting in 50 seconds...")
    time.sleep(50)
    start_websocket()

def refresh_jwt(ws):
    global jwt_token
    max_retries = 3
    retry_delay = 120  # seconds

    for attempt in range(max_retries):
        try:
            jwt_token = generate_jwt()
            print("refresh_jwt: JWT token refreshed")
            # Optionally, you can send a refresh message to the server
            ws.send(json.dumps({"type": "refresh", "jwt": jwt_token}))
            return  # Success, exit the function
        except Exception as e:
            if "Connection is already closed" in str(e):
                if attempt < max_retries - 1:  # If it's not the last attempt
                    print(f"Connection closed. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Error in refresh_jwt after {max_retries} attempts: {e}")
            else:
                print(f"Error in refresh_jwt: {e}")
                return  # Exit on other errors

def schedule_refresh(ws):
    try:
        refresh_jwt(ws)
        # Schedule the next refresh in 110 seconds (just before the 120-second expiration)
        threading.Timer(110, schedule_refresh, [ws]).start()
    except Exception as e:
        print(f"Error in schedule_refresh: {e}")


def on_open(ws):
    try:
        products = ["BTC-USDC", "ETH-USDC", "MATIC-USDC", "SOL-USDC", "DAI-USDC", "DOGE-USDC", "ETH-BTC", "MATIC-BTC", "SOL-BTC", "DOGE-BTC", "SOL-ETH", "ETH-DAI"]
        
        # Subscribe to ticker_batch channel
        subscribe_to_products(ws, products, CHANNEL_NAMES["ticker_batch"])
        print(f"on_open: Subscribed to ticker_batch for products: {products}")
        
        # Subscribe to heartbeats channel
        subscribe_to_products(ws, [], CHANNEL_NAMES["heartbeats"])
        #print(f"on_open: Subscribed to heartbeats")
        
        # Subscribe to user channel (requires authentication)
        #subscribe_to_products(ws, products+["MATIC-GBP"], CHANNEL_NAMES["user"])
        print(f"on_open: Subscribed to user channel for products: {products}")
      
        # Start the refresh cycle
        schedule_refresh(ws)
        
        # Optionally, add connection monitoring
        check_connection(ws)
        
    except Exception as e:
        print(f"Error in on_open: {e}")

# Initialize the last_non_heartbeat_time
last_non_heartbeat_time = time.time()

def check_connection(ws):
    global last_non_heartbeat_time
    try:
        if time.time() - last_non_heartbeat_time > 300:  # No non-heartbeat message for 5 minutes
            logging.warning("No non-heartbeat messages received for 5 minutes. Reconnecting...")
            ws.close()
        else:
            threading.Timer(60, lambda: check_connection(ws)).start()  # Check every minute
    except Exception as e:
        print(f"Error in check_connection: {e}")

def subscribe_to_products(ws, products, channel_name):
    try:
        message = {
            "type": "subscribe",
            "channel": channel_name,
            "product_ids": products,
            "jwt": jwt_token
        }
        ws.send(json.dumps(message))
        print(f"subscribe_to_products: Subscribed to {channel_name} for products: {products}")  # Debug statement
    except Exception as e:
        print(f"Error in subscribe_to_products: {e}")

def unsubscribe_to_products(ws, products, channel_name):
    try:
        message = {
            "type": "unsubscribe",
            "channel": channel_name,
            "product_ids": products
        }
        ws.send(json.dumps(message))
        print(f"unsubscribe_to_products: Unsubscribed from {channel_name} for products: {products}")  # Debug statement
    except Exception as e:
        print(f"Error in unsubscribe_to_products: {e}")

def on_message(ws, message):
    global last_non_heartbeat_time
    try:
        data = json.loads(message)
        
        if 'channel' in data:
            channel = data['channel']
            events = data.get('events', [])

            # Update last_non_heartbeat_time for non-heartbeat messages
            if channel != 'heartbeats':
                last_non_heartbeat_time = time.time()
            
            if channel == 'heartbeats':
                handle_heartbeat_event(data)
            
            elif channel == 'candles':
                for event in events:
                    if 'candles' in event:
                        handle_candle_event(event['candles'])
            
            elif channel == 'ticker_batch':
                timestamp = data['timestamp']
                for event in events:
                    if 'tickers' in event:
                        handle_ticker_batch_event(event['tickers'], timestamp)
            
            elif channel == 'user':
                print("on message user", data)
                handle_user_event(data)
    except Exception as e:
        print(f"Error in on_message: {e}")


def handle_user_event(data):
    print("handling user event", data)
    try:
        events = data['events']
        for event in events:
            if event['type'] == 'snapshot':
                orders = event['orders']
                for order in orders:
                    print(f"handle_user_event: processing order {order}")  # Debug statement
                    # Process the order data as needed
                    # Example: produce to Kafka or log the order details
                    producer.produce('crypto_user_orders', json.dumps(order))
        producer.flush()
    except Exception as e:
        print(f"Error in handle_user_event: {e}")
        logging.error(f"Error in handle_user_event: {e}")

def handle_candle_event(candles):
    try:
        for candle in candles:
            print(f"handle_candle_event: processing candle {candle}")  # Debug statement
            df = pd.DataFrame([candle], columns=['start', 'high', 'low', 'open', 'close', 'volume', 'product_id'])
            df['start'] = pd.to_datetime(pd.to_numeric(df['start']), unit='s')
            df.set_index('start', inplace=True)
            for index, row in df.iterrows():
                producer.produce('crypto_candles', json.dumps({
                    'start': index.isoformat(),
                    'product_id': row['product_id'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }))
            producer.flush()
    except Exception as e:
        print(f"Error in handle_candle_event: {e}")

def handle_heartbeat_event(heartbeat):
    try:
        events = heartbeat['events'][0]
        counter = events['heartbeat_counter']
        print(f"HEARTBEAT {counter}")  # Debug statement            
    except Exception as e:
        print(f"Error in handle_heartbeat_event: {e}")

def handle_ticker_batch_event(tickers, timestamp):
    try:
        for ticker in tickers:
            ticker['time'] = pd.to_datetime(timestamp).isoformat()
            print(f"handle_ticker_batch_event: {ticker}")  # Debug statement
            producer.produce('crypto_ticker_batch', json.dumps(ticker))
            producer.flush()
    except Exception as e:
        print(f"Error in handle_ticker_batch_event: {e}")

def start_websocket():
    try:
        print("start_websocket: starting websocket connection")
        ws = websocket.WebSocketApp(WS_API_URL, 
                                    on_open=on_open, 
                                    on_message=on_message, 
                                    on_error=on_error, 
                                    on_close=on_close)
        ws.run_forever(sslopt={"ca_certs": certifi.where()})
    except Exception as e:
        print(f"Error in start_websocket: {e}")

class DataLoader:
    def __init__(self):
        try:
            print("DataLoader.__init__: initializing DataLoader")
            self.cryptos = ["ETH", "USDC", "MATIC", "SOL", "DAI", "BTC", "DOGE"]
            self.valid_pairs = ["ETH-USDC", "BTC-USD", "ETH-USD", "BTC-ETH"]
            self.pairs = [pair for pair in self.valid_pairs if pair.split('-')[0] in self.cryptos and pair.split('-')[1] in self.cryptos]
            self.resolution = "minute"  # Set resolution to minute
            self.granularity = 60  # Coinbase uses seconds, so 60 for a minute
        except Exception as e:
            print(f"Error in DataLoader.__init__: {e}")

    def load_data(self):
        try:
            print("DataLoader.load_data: loading data")
            start_websocket()
        except Exception as e:
            print(f"Error in DataLoader.load_data: {e}")

if __name__ == "__main__":
    try:
        DataLoader().load_data()
    except Exception as e:
        print(f"Error in __main__: {e}")