from model.sequential import Sequential
from model.util.layer import Layer
import numpy as np

x = [[0.1, 0.9]]
y = [[0.9]]

model = Sequential([
  Layer(2),
  Layer(1),
])

model.compile()
model.fit(
  x,
  y,
  epochs=1,
  batch_size=1
)