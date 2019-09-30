# Import
from model.sequential import Sequential
from model.util.layer import Layer

from util.preprocess import preprocess
from util.metrics import binary_accuracy_score

import matplotlib.pyplot as plt

# Read *.csv file and preprocess
X, y = preprocess('./test/data_weather.csv')

# Define nb of layers and nb of neurons
model = Sequential([
  Layer(5), # param 1: nb of neurons
  Layer(5),
  Layer(5),
  Layer(5),
  Layer(1)
])

# define batch_size and epochs as you wish!
batch_size = 2
epochs = 1000

# build model
model.compile()

# let's train!
model.fit(
  X,
  y,
  epochs=epochs,
  batch_size=batch_size,
)

# plotting cost (error)
times = [i for i in range(epochs)]

fig, ax = plt.subplots(figsize=(5,3))
ax.plot(times, model.optimizer.error_list)
fig.suptitle('Plotting Error')

ax.set_xlabel('Iterations / Epochs')
ax.set_ylabel('Error')

# show the plot
plt.show()

# predict input data by using predict method
y_pred = model.predict(X)

print('\n\nerror: {}'.format(model.optimizer.error_list.pop()))
print('output predict {}:\n{}'.format(X, y_pred))
print('accuracy: {}'.format(binary_accuracy_score(y, y_pred)))