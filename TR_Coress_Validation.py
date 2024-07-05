from timeconvert import TimeConvert
from selecttimetodelete import SelectTimeToDeleteOptimized
from preparing_data import PREPARE_DATA
from PAGECREATOR import PageCreatorParallel
from deleterow import DeleteRow
from FEATURESELECTION import FeatureSelection
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

class CrossValidation:

    def normalize_data(self, data):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        return data_scaled


    def Train(self, data, depth, page, feature, QTY, iter, Thereshhold, primit_hours=[]):
        try :
            print(f"depth:{depth}, page:{page}, features:{feature}, QTY:{QTY}, iter:{iter}, Thereshhold:{Thereshhold}, primit_hours:{primit_hours}")
            data = data[-QTY:]
            data = TimeConvert().exec(data)        
            data.reset_index(inplace=True, drop=True)
            primit_hours = SelectTimeToDeleteOptimized().exec(data, primit_hours)
            data, target, primit_hours = PREPARE_DATA().ready(data, primit_hours)
            data, target = PageCreatorParallel().create_dataset(data, target, page)
            primit_hours = primit_hours[page:]
            data, target = DeleteRow().exec(data, target, primit_hours)
            fs = FeatureSelection()
            selected_data, selected_features = fs.select(data, target, feature)
            data = selected_data.copy()
            data = self.normalize_data(data)
            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=1234)
        except :
            print("Error in preparing data .... program rised")
            return(0, )
        # print(f"primit_hours  = {primit_hours}")
        if iter < 1000 :
            iter = 1000
        model = CatBoostClassifier(
            iterations=iter,
            learning_rate=0.01,
            depth=depth,
            loss_function='Logloss',
            eval_metric='AUC',
            verbose=50,
            task_type='CPU',
            random_state=42
        )

        try :
            print("Start coross valiation ...")
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
            print("k_fold ACC = ", scores)
            print("k_fold ACC mean = ", np.mean(scores))
            print("End of coross valiation ...")
            return (np.mean(scores), )
        except :
            print("K_fold fild .... program rised")
            return (0, )
