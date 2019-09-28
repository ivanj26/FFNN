import numpy as np

class Node:
  def __init__(self, activation, bias=np.random.rand()):
    self.bias = bias
    self.weights = []
    self.delta_weights = []
    self.temp_delta_weights = []
    self.delta_bias = 0
    self.temp_delta_bias = 0
    self.net = 0
    self.gradient = 0
    self.activation = activation
  
  def rand_weights(self, dimension: int):
    for _ in range(dimension):
      self.weights.append(np.random.rand())
      print(len(self.weights))
      self.delta_weights.append(0)
      self.temp_delta_weights.append(0)   

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

  def calc_delta_weights(self, learning_rate, momentum, x = []):
    for i in range(len(self.delta_weights)):
      self.delta_weights[i] = learning_rate * self.gradient * x[i] + momentum * self.delta_weights[i]
      self.temp_delta_weights[i] = self.delta_weights[i]
    return self.delta_weights

  def calc_delta_bias(self, learning_rate, momentum):
    self.delta_bias = learning_rate * self.gradient + momentum * self.delta_bias
    self.temp_delta_bias = self.delta_bias
    return self.delta_bias

  def set_weights(self, w):
    self.weights = w

  def sum_delta_and_weights(self):
    for i in range(len(self.weights)):
      self.weights[i] += self.delta_weights[i]
      self.delta_weights[i] = 0
      self.temp_delta_weights[i] = 0

  def sum_delta_and_bias(self):
    self.bias += self.delta_bias
    self.delta_bias = 0
    self.temp_delta_bias = 0