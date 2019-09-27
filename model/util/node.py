import numpy as np

class Node:
  def __init__(self, activation, bias):
    self.delta_weights = []
    self.bias = np.random.rand()
    self.bias = bias
    self.weights = []
    self.net = 0
    self.gradient = 0
    self.activation = activation
  
  def rand_weights(self, dimension: int):
    for _ in range(dimension):
      self.weights.append(np.random.rand())

  def calc_dot_prod(self, x = []):
    sum = 0
    for i in range(len(x)):
      sum += (x[i] * self.weights[i])

    return sum + self.bias

  def calc_net(self, x = []):
    self.net = self.activation(self.calc_dot_prod(x))
    return self.net

  def calc_gradient(self, i, next_weights = [], next_gradients = []):
    sum_gradient = 0
    for j in range(len(next_gradients)):
      sum_gradient += next_weights[j][i] * next_gradients[j]
    self.gradient = self.net * (1 - self.net) * sum_gradient
    return self.gradient

  def set_weights(self, w):
    self.weights = w