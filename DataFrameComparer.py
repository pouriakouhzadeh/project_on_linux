import pandas as pd
import numpy as np

class DataFrameComparer:
    def __init__(self, df1, df2):
        self.df1 = df1
        self.df2 = df2
        self.differences = None
        self.percent_changed = 0
    
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
        
        if self.percent_changed > 8:
            return True
            # اینجا می‌توانید کد لازم برای بازآموزی مدل یا سایر اقدامات را قرار دهید
        else:
            return False

# نمونه استفاده از کلاس:
# df1 = pd.DataFrame(...) # دیتافریم اول
# df2 = pd.DataFrame(...) # دیتافریم دوم

# comparer = DataFrameComparer(df1, df2)
# comparer.compare_dataframes()
# comparer.report_changes()
