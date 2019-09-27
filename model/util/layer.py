from model.util.activation_fun import sigmoid
from model.util.node import Node
import numpy as np

class Layer:
  def __init__(self, nb_neurons = 1, activation = 'sigmoid'):
    self.nb_neurons = nb_neurons
    self.activation = None

    if (activation == 'sigmoid'):
      self.activation = sigmoid
    
    self.neurons = [Node(self.activation) for i in range(self.nb_neurons)]

  def feed_forward(self, x_before = []):
    for i in range(self.nb_neurons):
      self.neurons[i].calc_net(x_before)
    return [n.net for n in self.neurons]
  
  def back_propagate(self, weights_after, deltas_after):
    sum_deltas = 0
    for weight, delta in weights_after, deltas_after:
      sum_deltas += delta * weight
    for neuron in self.neurons:
      neuron.calc_error(sum_deltas)
    return np.asarray([n.weights for n in self.neurons]), np.asarray([n.delta for n in self.neurons])