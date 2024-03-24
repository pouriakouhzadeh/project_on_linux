from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

class FeatureSelection:
    def select(self, data, target, n, **rf_params):
        # Ensure `n` is an integer
        n = int(n)
        
        # Convert data to a DataFrame if not already one
        data = pd.DataFrame(data)
        
        # Initialize and fit the RandomForestClassifier
        model = RandomForestClassifier(**rf_params, n_jobs=-1)
        model.fit(data, target)
        
        # Get feature importances and sort them in descending order
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Select the top `n` features' indices
        selected_indices = indices[:n]
        
        # Select the top `n` features from the data
        selected_features = data.columns[selected_indices]
        X_selected = data[selected_features]
        
        return pd.DataFrame(X_selected, columns=selected_features), selected_features

# Example usage:
# Assuming `data` is a DataFrame and `target` is the target column
# fs = FeatureSelection()
# X_selected, selected_features = fs.select(data, target, n=5, n_estimators=100, max_depth=5)
# print("Selected features:", selected_features)
