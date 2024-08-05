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
