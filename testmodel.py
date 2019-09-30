from model.sequential import Sequential
from model.util.layer import Layer
from model.util.node import Node
from model.util.activation_fun import sigmoid

import matplotlib.pyplot as plt
from util.preprocess import preprocess
from util.metrics import binary_accuracy_score
import numpy as np

# data dari slide kuliah A.I
x = [[0.1, 0.9]]
y = [[0.9]]
epochs = 1000

model = Sequential([
  Layer(2),
  Layer(1),
])

model.compile()
model.fit(
  x,
  y,
  epochs=epochs,
  batch_size=1
)

# plotting cost (error)
times = [i for i in range(len(model.optimizer.error_list))]

plt.plot(times, model.optimizer.error_list)
plt.show()

print('\n\nerror: {}'.format(model.optimizer.error_list.pop()))
print('output predict {}: {}'.format(x[0], model.predict([x[0]])))