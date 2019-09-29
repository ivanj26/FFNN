import pandas as pd
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

def read_file(path):
  df_weather = pd.read_csv(path)
  return df_weather.drop(['play'], axis=1).values, df_weather['play'].values
  
X, y = read_file('../test/data_weather.csv')
X[:, 0], X[:, 3] = encoder.fit_transform(X[:, 0]), encoder.fit_transform(X[:, 3])
y = encoder.fit_transform(y)

print(X)
print(y)