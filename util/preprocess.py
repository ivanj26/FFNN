from sklearn.preprocessing import LabelEncoder
from util.read_file import read_file

def preprocess(path):
  X, y = read_file(path)

  encoder = LabelEncoder()

  X[:, 0], X[:, 3] = encoder.fit_transform(X[:, 0]), encoder.fit_transform(X[:, 3])
  y = encoder.fit_transform(y)

  X = X.tolist()
  y = y.tolist()

  y = [[e] for e in y]
  
  return X, y