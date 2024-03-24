import pandas as pd
import numpy as np

class SelectTimeToDeleteOptimized:
    
    def exec(self, data, n=[]):


        if not isinstance(n, (list, pd.Series, np.ndarray, set)):
            n = [n]  # تبدیل `n` به یک لیست اگر لیستی نیست
        # ایجاد یک سری بولی برای شناسایی ردیف‌هایی که ساعت آنها در لیست `n` قرار دارد
        condition = data["Hour"].isin(n)
        
        # استفاده از np.where برای ایجاد یک ستون جدید بر اساس شرط: علامت‌گذاری با 1 برای شرایط درست و NaN برای شرایط نادرست
        Forbidden = pd.DataFrame(np.where(condition, 1, np.nan), columns=['Marked'])
        
        return Forbidden
