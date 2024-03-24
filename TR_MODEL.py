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

class TrainModels:

    def normalize_data(self, data):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        return data_scaled


    def ACC_BY_THRESHHOLD(self, y_test, predictions_proba, TH):
        predictions_proba = pd.DataFrame(predictions_proba)
        predictions_proba.reset_index(inplace = True ,drop =True)
        y_test.reset_index(inplace = True ,drop =True)
        TH = TH / 100
        try :
            wins = 0
            loses = 0
            for i in range(len(y_test)) :
                if predictions_proba[1][i] > TH :
                    if y_test['close'][i] == 1 :
                        wins = wins + 1
                    else :
                        loses = loses + 1    
                if predictions_proba[0][i] > TH :
                    if y_test['close'][i] == 0 :
                        wins = wins + 1
                    else :
                        loses = loses + 1       
            # logging.info(f"Thereshhold wins = {wins}, Thereshhold loses = {loses}")
            return ( (wins * 100) / (wins + loses) , wins, loses)  
        except :
            return 0, 0, 0

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
            iterations=iter, # شروع با تعداد زیادی تکرار
            learning_rate=0.01,
            depth=depth,
            l2_leaf_reg=5,
            loss_function='Logloss',
            verbose=False,
            task_type='CPU',
            random_state=42
        )

        # try :
        #     print("Start coross valiation ...")
        #     kf = KFold(n_splits=5, shuffle=True, random_state=42)
        #     scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
        #     print("k_fold ACC = ", scores)
        #     print("k_fold ACC mean = ", np.mean(scores))
        #     print("End of coross valiation ...")
        #     if np.mean(scores) < 0.65 :
        #         print("K_fold not accept this parametes .... program rised")
        #         return (0, )
        # except :
        #     print("K_fold fild .... program rised")
        #     return (0, )
        try :
            print("Start training model")
            model.fit(X_train, y_train)
            print("End of training model")
            predictions_proba = model.predict_proba(X_test)
            ACC , wins , loses = self.ACC_BY_THRESHHOLD(y_test, predictions_proba, Thereshhold)
            print(f"ACC:{ACC}, wins:{wins}, loses:{loses}")
        except :
            print("Error in training model ... program rised")
            return(0, )

        try :
            if wins + loses >= 0.2 * ((QTY * 100)/100):  # این شرط باید بر اساس لاجیک مورد نظر شما تنظیم شود
                return (ACC/100, )
            else:
                return (0, )
        except Exception as e:
            print(f"Error processing task: {e}")
            return (0, )
