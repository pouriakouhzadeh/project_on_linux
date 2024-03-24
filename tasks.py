from celery_app import app
from TR_MODEL import TrainModels
import pandas as pd
import json

TR = TrainModels()

@app.task
def json_to_df(json_data):
    return pd.read_json(json_data, orient='split')

@app.task(bind=True)
def train_model_task(self, currency_data, depth, page, feature, QTY, iter, Thereshhold, primit_hours):
    currency_data_df = json_to_df(currency_data)

    print("New job received, starting to process...")
    acc = TR.Train(currency_data_df, depth, page, feature, QTY, iter, Thereshhold, primit_hours)
    return acc
