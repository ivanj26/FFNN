from model.sequential import Sequential
from model.util.layer import Layer
from model.util.node import Node
from model.util.activation_fun import sigmoid
import numpy as np

x = [[0.1, 0.9]]
y = [[0.9]]

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
model = Sequential([
  Layer(2),
  Layer(1),
])

model.compile()
model.fit(
  x,
  y,
  epochs=1000,
  batch_size=1
)

print('\n\noutput predict {}: {}'.format(x[0], model.predict(x)))