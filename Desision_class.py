from timeconvert import TimeConvert
from selecttimetodelete import SelectTimeToDeleteOptimized
from preparing_data import PREPARE_DATA
from PAGECREATOR import PageCreatorParallel
from deleterow import DeleteRow
from FEATURESELECTION_FOR_DESISION import FeatureSelection
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
import numpy as np
logging.basicConfig(filename="Posittion_history.log", level=logging.INFO)

def normalize_data(data, currency):
    with open(f"/home/pouria/project/trained_models/{currency}60_scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    data_scaled = scaler.transform(data)
    return data_scaled

class DesisionClass:


    def Desision(self, data, currency, ind , primit_houre):
        fs = FeatureSelection()
        position = "NAN"
        Answer = [0,0]
        current_houre = pd.to_datetime(data.iloc[-1,0]).hour  
        
        if current_houre in primit_houre :
            
            print("Primit houre is allowed ...")
            page = ind[1]
            features = ind[2]
            thereshhold = ind[-1]
            QTY = ind[3]

            data = data[-QTY:]
            data = TimeConvert().exec(data)        
            data.reset_index(inplace=True, drop=True)
            primit_hours = SelectTimeToDeleteOptimized().exec(data, primit_houre)
            data, target, primit_hours = PREPARE_DATA().ready(data, primit_hours)
            data, target = PageCreatorParallel().create_dataset(data, target, page)
            primit_hours = primit_hours[page:]
            data, target = DeleteRow().exec(data, target, primit_hours)
            selected_data = fs.select(data, currency)
            data = selected_data.copy()
            data = np.array(data)
            data = normalize_data(data, currency)

            with open(f"/home/pouria/project/trained_models/{currency}60.pkl", 'rb') as f:
                model = pickle.load(f)

            predict = model.predict_proba(data[-1])

            if predict[0] > (thereshhold/100) :
                position = "SELL"
            if predict[1] > (thereshhold/100) :
                position = "BUY"            
            Answer = predict

            if position == "BUY" or position == "SELL" :
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                logging.info(f"Position : {position}, Market : {currency}, Date & Time : {dt_string}, Answer : {Answer}, TH : {thereshhold/100}")
            return position, Answer
            
        else:
            print("Cannot trade in this time so return")
            position = "FBT"
            return position, Answer  

