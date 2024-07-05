import pandas as pd
import numpy as np

class DataFrameComparer:
    def __init__(self, df1, df2):
        self.df1 = df1
        self.df2 = df2
        self.differences = None
        self.percent_changed = 0
    
    def My_compare_dataframes(self):
        lenth = len(self.df1)
        first_time = self.df1.iloc[-1,0]
        second_time = self.df2.iloc[-1,0]
        first_time = pd.to_datetime(first_time)
        second_time = pd.to_datetime(second_time)
        difrence_time = first_time - second_time
        difrence_time = difrence_time.total_seconds() / 3600
        return np.abs((difrence_time / lenth) * 100)


    def compare_dataframes(self):
        """مقایسه دو دیتافریم و محاسبه درصد تغییرات."""
        self.differences = self.df1 != self.df2
        changed_cells = np.sum(self.differences.values)
        total_cells = self.differences.size
        self.percent_changed = (changed_cells / total_cells) * 100
    
    def report_changes(self):
        """گزارش درصد تغییرات و چک کردن برای آستانه مشخص (8%)."""
        print(f"number of chenged data : {np.sum(self.differences.values)}")
        print(f"Total cells : {self.differences.size}")
        print(f"Exchange rate : {self.percent_changed}%")
        
        return self.percent_changed 



# نمونه استفاده از کلاس:
# df1 = pd.DataFrame(...) # دیتافریم اول
# df2 = pd.DataFrame(...) # دیتافریم دوم

# comparer = DataFrameComparer(df1, df2)
# comparer.compare_dataframes()
# comparer.report_changes()
