from util.activation_fun import sigmoid
from node import Node

class Layer:
  def __init__(self, nb_neurons = 1, activation = 'sigmoid'):
    self.nb_neurons = nb_neurons
    self.neurons = [Node(i) for i in range(self.nb_neurons)]
    if (activation == 'sigmoid'):
      self.activation = sigmoid