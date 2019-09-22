import numpy as np

class Node:
  def __init__(self, activation):
    self.bias = np.random.rand()
    self.weights = []
    self.activation = activation
  
  def rand_weights(self, dimension: int):
    for _ in range(dimension):
      self.weights.append(np.random.rand())

  def calc_dot_prod(self, x):
    sum = 0
    for i in range(len(x)):
      sum += (x[i] * self.weights[i])

    return sum + self.bias

  def calc_net(self, x):
    return self.activation(self.calc_dot_prod(x))