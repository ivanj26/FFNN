from model.optimizer.minibatchgd import MiniBatchGD

class Sequential:
  def __init__(self, layers = []):
    self.layers = layers
    self.optimizer = None

  def add(self, layer):
    self.layers.append(layer)

  def compile(self, optimizer = 'MiniBatchGD', lr = 0.25, momentum = 0.0001):
    opt = None

    if (optimizer == 'MiniBatchGD'):
      opt = MiniBatchGD(self.layers, lr = lr, momentum = momentum)

    if (opt == None):
      print('Failed build model!')

    self.optimizer = opt

  def fit(self, x, y, epochs = 10000, batch_size = 32):
    self.optimizer.fit(x, y, epochs, batch_size)

  def predict(self, x):
    self.optimizer.predict(x)