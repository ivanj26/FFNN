from model.sequential import Sequential
from model.util.layer import Layer
from model.util.activation_fun import sigmoid
from model.util.node import Node
import numpy as np

n11 = Node(sigmoid, 0.1)
n11.set_weights([-0.2, 0.1])
n11.init_delta_weights()

n12 = Node(sigmoid, 0.1)
n12.set_weights([-0.1, 0.3])
n12.init_delta_weights()
n1 = [n11, n12]

l1 = Layer(2, neurons=n1)

# layer 2

n21 = Node(sigmoid, 0.2)
n21.set_weights([0.2, 0.3])
n21.init_delta_weights()
n2 = [n21]

l2 = Layer(1, neurons=n2)

model = Sequential([
  l1,
  l2
])

model.compile()

x = [[0.1, 0.9]]
y = [[0.9]]
model.fit(
  x,
  y,
  10,
  1,
)
