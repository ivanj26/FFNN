class MiniBatchGD:
  def __init__(self, layers, lr, momentum):
    self.lr = lr
    self.momentum = momentum
    self.layers = layers
  
  def fit(self, x, y, epochs, batch_size):
    pass

  def predict(self, x):
    pass