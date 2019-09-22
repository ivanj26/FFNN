from model.util.activation_fun import sigmoid
from model.node import Node

class Layer:
  def __init__(self, nb_neurons = 1, activation = 'sigmoid'):
    self.nb_neurons = nb_neurons
    self.activation = None

    if (activation == 'sigmoid'):
      self.activation = sigmoid
    
    self.neurons = [Node(self.activation) for i in range(self.nb_neurons)]