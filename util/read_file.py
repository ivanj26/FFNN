import pandas as pd
from sklearn.preprocessing import LabelEncoder

def read_file(path):
  df_weather = pd.read_csv(path)
  return df_weather.drop(['play'], axis=1).values, df_weather['play'].values