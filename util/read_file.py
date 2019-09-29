import pandas as pd

def read_file(path):
  df_weather = pd.read_csv(path)
  return df_weather.drop(['play'], axis=1).values, df_weather['play'].values
  
X, y = read_file('../test/data_weather.csv')
print(X)