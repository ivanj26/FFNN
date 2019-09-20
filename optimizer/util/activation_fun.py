import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
  """
  Calculate net using sigmoid activation
  :type x: int
  :param x: dot product

  """
  return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
  """
  Calculate derivative of sigmoid
  :type x: int
  :param x: dot product

  """
  s = sigmoid(x)
  return s * (1 - s)