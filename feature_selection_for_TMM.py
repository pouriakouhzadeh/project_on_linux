import pandas as pd

class FeatureSelection_for_TMM:
    def select(self, data, target, num_features, model):
        model.fit(data, target)
        feature_importances = model.get_feature_importance()
        features = data.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        selected_features = importance_df.head(num_features)['Feature']
        selected_columns = [data.columns.get_loc(feature) for feature in selected_features]

        return selected_features, selected_columns
