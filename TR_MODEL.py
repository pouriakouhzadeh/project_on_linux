import logging
from timeconvert import TimeConvert
from selecttimetodelete import SelectTimeToDeleteOptimized
from preparing_data_for_train import PREPARE_DATA_FOR_TRAIN
from PAGECREATOR import PageCreatorParallel
from deleterow import DeleteRow
from feature_selection_for_TMM import FeatureSelection_for_TMM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import gc
from sklearn.metrics import f1_score, confusion_matrix
import math
from CREATEMODEL import CreateModel

# تنظیمات logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    handlers=[
                        logging.FileHandler("train_models.log"),
                        logging.StreamHandler()
                    ])

class TrainModels:

    def normalize_data(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def calculate_balanced_accuracy(self, y_true, predictions_proba, threshold):
        y_true_filtered = []
        y_pred_filtered = []
        threshold = threshold / 100
        for i in range(len(y_true)):
            if predictions_proba[i][1] > threshold:
                y_pred_filtered.append(1)
                y_true_filtered.append(y_true[i])
            elif predictions_proba[i][0] > threshold:
                y_pred_filtered.append(0)
                y_true_filtered.append(y_true[i])

        if not y_true_filtered:  # اطمینان حاصل کنید که لیست خالی نیست
            return 0.5  # مقدار متعادل پیش‌فرض

        try:
            cm = confusion_matrix(y_true_filtered, y_pred_filtered)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                balanced_accuracy = (sensitivity + specificity) / 2
                if math.isnan(balanced_accuracy):
                    balanced_accuracy = 0.0
                return balanced_accuracy
            elif cm.shape == (1, 2) or cm.shape == (2, 1):
                print("Only one class present in y_true. Returning default balanced accuracy of 0.5.")
                return 0.5  # مقدار متعادل پیش‌فرض
            else:
                print(f"Confusion matrix shape is not valid: {cm.shape}")
                return 0.5  # مقدار متعادل پیش‌فرض
        except Exception as e:
            print(f"Error in calculate_balanced_accuracy: {e}")
            return 0.5  # مقدار متعادل پیش‌فرض

    def F1_score(self, y_test, predictions_proba, threshold):
        predictions_proba = pd.DataFrame(predictions_proba)
        y_test = pd.Series(y_test)  # تبدیل y_test به سری برای دسترسی آسان‌تر

        predictions_proba.reset_index(inplace=True, drop=True)
        y_test.reset_index(drop=True, inplace=True)
        threshold = threshold / 100

        try:
            predictions = []
            y_test_filtered = []
            for i in range(len(y_test)):
                if predictions_proba.iloc[i, 1] > threshold:
                    predictions.append(1)
                    y_test_filtered.append(y_test[i])
                elif predictions_proba.iloc[i, 0] > threshold:
                    predictions.append(0)
                    y_test_filtered.append(y_test[i])

            if len(predictions) == 0:
                return 0, 0, 0, 0

            f1 = f1_score(y_test_filtered, predictions)
            wins = sum(1 for i in range(len(y_test_filtered)) if y_test_filtered[i] == predictions[i])
            loses = len(y_test_filtered) - wins
            acc = (wins * 100) / (wins + loses)
            return f1, wins, loses, acc
        except Exception as e:
            logging.error(f"Error in F1_score: {e}")
            return 0, 0, 0, 0

    def Train(self, data, depth, page, feature, QTY, iterations, Thereshhold, primit_hours=[]):
        fs = FeatureSelection_for_TMM()
        model = CreateModel.create(data, iterations, depth)
        try:
            print(f"depth:{depth}, page:{page}, features:{feature}, QTY:{QTY}, iter:{iterations}, Thereshhold:{Thereshhold}, primit_hours:{primit_hours}")
            data = data[-QTY:]
            data = TimeConvert().exec(data)
            data.reset_index(inplace=True, drop=True)
            primit_hours = SelectTimeToDeleteOptimized().exec(data, primit_hours)
            data, target, primit_hours = PREPARE_DATA_FOR_TRAIN().ready(data, primit_hours)
            data, target = PageCreatorParallel().create_dataset(data, target, page)
            primit_hours = primit_hours[page:]
            data, target = DeleteRow().exec(data, target, primit_hours)
            selected_data ,selected_columns = fs.select(data, target, feature, model)
            data = selected_data.copy()
            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=1234)

            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            y_train = y_train.reshape(-1)
            y_test = y_test.reshape(-1)

            X_train, X_test = self.normalize_data(X_train, X_test)
        except Exception as e:
            print(f"Error in preparing data: {e}")
            return 0, 0, 0, 0

        if iterations < 1000:
            iterations = 1000



        try:
            logging.info(f"Start training model with depth = {depth}, pages = {page}, iteration = {iterations}")
            model.fit(X_train, y_train, eval_set=(X_test, y_test))
            logging.info("End of training model")
            predictions_proba = model.predict_proba(X_test)
            f1, wins, loses, acc = self.F1_score(y_test, predictions_proba, Thereshhold)
            balanced_acc = self.calculate_balanced_accuracy(y_test, predictions_proba, Thereshhold)
            logging.info(f"F1_score: {f1}, ACC:{acc}, Balanced_acc = {balanced_acc}, wins: {wins}, loses: {loses}")
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            del data, selected_data, X_train, X_test, y_train, target
            gc.collect()
            return 0, 0

        try:
            if wins + loses >= 0.2 * ((QTY * 10) / 100):
                del data, selected_data, X_train, X_test, y_train, target
                gc.collect()
                if f1 > 0.5 and balanced_acc > 0.5 :    
                    return f1, balanced_acc
                else :
                    return 0, 0
            else:
                del data, selected_data, X_train, X_test, y_train, target
                gc.collect()
                return 0, 0
        except Exception as e:
            logging.error(f"Error processing task: {e}")
            del data, selected_data, X_train, X_test, y_train, target
            gc.collect()
            return 0, 0