from celery_app import app
from TR_MODEL import TrainModels
import pandas as pd
import json
import base64
import gzip
import logging
from io import BytesIO

TR = TrainModels()

# Configure logging to save logs to a file and display them in the console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    handlers=[
                        logging.FileHandler("tasks.log"),
                        logging.StreamHandler()
                    ])

def decode_and_decompress(encoded_gz_data):
    try:
        # Decode the base64 encoded gzip data
        gz_data = base64.b64decode(encoded_gz_data)
        
        # Decompress the gzip data in memory
        with gzip.GzipFile(fileobj=BytesIO(gz_data), mode='rb') as f:
            json_data = f.read().decode('utf-8')
        
        return json.loads(json_data)
    except Exception as e:
        logging.error(f"Error decoding and decompressing the JSON file: {e}")
        raise

@app.task
def json_to_df(json_data):
    try:
        return pd.read_json(json.dumps(json_data), orient='split')
    except Exception as e:
        logging.error(f"Error converting JSON to DataFrame: {e}")
        raise

@app.task(bind=True)
def train_model_task(self, encoded_gz_data, depth, page, feature, QTY, iter, Thereshhold, primit_hours):
    try:
        # Decode and decompress the received data
        json_data = decode_and_decompress(encoded_gz_data)
        
        # Convert JSON to DataFrame
        currency_data_df = json_to_df(json_data)
        
        logging.info("New job received, starting to process...")
        
        # Train the model
        acc = TR.Train(currency_data_df, depth, page, feature, QTY, iter, Thereshhold, primit_hours)
        
        return acc
    except Exception as e:
        logging.error(f"Error in train_model_task: {e}")
        self.retry(exc=e)
