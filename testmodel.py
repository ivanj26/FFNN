from model.sequential import Sequential
from model.util.layer import Layer
from model.util.node import Node
from model.util.activation_fun import sigmoid

import matplotlib.pyplot as plt
from util.preprocess import preprocess
import numpy as np

x = [[0.1, 0.9]]
y = [[0.9]]
epochs = 1000

X, y = preprocess('./test/data_weather.csv')

model = Sequential([
  Layer(2),
  Layer(2),
  Layer(2),
  Layer(2),
  Layer(1),
])

model.fit(
  X,
  y,
  epochs=epochs,
  batch_size=1
)

# n11 = Node(sigmoid, [-0.2, 0.1], 0.1)
# n12 = Node(sigmoid, [-0.1, 0.3], 0.1)

# # 2 node di layer 1
# n1 = [n11, n12]

# # layer 2
# n21 = Node(sigmoid, [0.2, 0.3], 0.2)
# n2 = [n21]

# model = Sequential([
#   Layer(2, neurons=n1),
#   Layer(1, neurons=n2),
# ])

# versi weights random
# model = Sequential([
#   Layer(2),
#   Layer(1),
# ])

# model.compile()
# model.fit(
#   x,
#   y,
#   epochs=epochs,
#   batch_size=1
# )

# plotting cost (error)
times = [i for i in range(len(model.optimizer.error_list))]

plt.plot(times, model.optimizer.error_list)
plt.show()

print('\n\nerror: {}'.format(model.optimizer.error_list.pop()))
print('output predict {}: {}'.format(X[0], model.predict([X[0]])))