from model.sequential import Sequential
from model.util.layer import Layer
from model.util.activation_fun import sigmoid
import numpy as np

n11 = Node(sigmoid, 0.1)
n11.set_weights([-0.2, 0.1])

n12 = Node(sigmoid, 0.1)
n12.set_weights([-0.1, 0.3])

n1 = [n11, n12]

l1 = Layer(2, n1)

# layer 2

n21 = Node(sigmoid, 0.2)
n21.set_weights([0.2, 0.3])
n2 = [n21]

l2 = Layer(2, n2)

model = Sequential([
  l1,
  l2
])

x = [[1.0, 0.1, 0.9, 1.0]]
y = [[0.9]]
model.fit(
  x,
  y,
  1,
  1,
)