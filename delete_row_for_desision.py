import pandas as pd
from tqdm import tqdm

class DeleteRow_for_desision:
    
    def exec(self, data, primit_hours):
        data = pd.DataFrame(data)
        data.reset_index(inplace = True, drop = True)
        data['primit_hours'] = primit_hours
        data.dropna(inplace=True)
        data.drop(columns = 'primit_hours', inplace = True)
        data.reset_index(inplace= True, drop = True)

        return data