from catboost import CatBoostClassifier

class CreateModel : 
    def create(self, iterations, depth) :
        model = CatBoostClassifier(
                iterations=iterations,
                learning_rate=0.005,
                depth=depth,
                loss_function='Logloss',
                verbose=100,
                task_type='CPU',
                random_state=42,
                eval_metric='F1',
                early_stopping_rounds=500,
                l2_leaf_reg=5,
                subsample=0.9,
                bagging_temperature=1,
            )
        return model