from TR_MODEL import TrainModels
import pandas as pd


def read_data(file_name, tail_size=21000):
    file_path = f"/home/pouria/project/temp_csv_dir/{file_name}"
    try:
        data = pd.read_csv(file_path)
        data = data.tail(tail_size)
        data.reset_index(inplace=True, drop=True)
        return data
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return None

currency_pair = "GBPUSD60"  
currency_data = read_data(f"{currency_pair}.csv", 21000)
TRMM = TrainModels()
# [(2, 12), (2, 30), (30, 800), (10000, 20000), (1000, 3000), (52, 75), (8, 18)]
model, feature_indicts, scaler , acc= TRMM.Train(currency_data.copy(), 2, 10, 100, 10000, 200, 52, [8,10,12,14,16,20,18])
print(feature_indicts)