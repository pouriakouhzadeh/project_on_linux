from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE

class FeatureSelection_for_TMM:
    def select(self, data, target, num_features, model):
        rfe = RFE(model, n_features_to_select=num_features)
        rfe = rfe.fit(data, target)
        selected_features = data.columns[rfe.support_]
        selected_columns = [i for i, x in enumerate(rfe.support_) if x]
        return selected_features, selected_columns
