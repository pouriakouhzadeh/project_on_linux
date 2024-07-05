import pandas as pd
import pickle

class FeatureSelection:
    def select(self, data, currency ):
        with open(f"/home/pouria/project/trained_models/{currency}60_features_indicts.pkl", 'rb') as f:
            selected_columns = pickle.load(f)

        X_selected = data.iloc[:, selected_columns]
        
        return X_selected

