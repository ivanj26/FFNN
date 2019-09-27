import numpy as np

class MiniBatchGD:
  def __init__(self, layers, lr, momentum):
    self.lr = lr
    self.momentum = momentum
    self.layers = layers
  
  def fit(self, x, y, epochs, batch_size):
    self.init_weights(x)

  def predict(self, x):
    pass

  def init_weights(self, x):
    dim = len(x[0])

    for layer in self.layers:
      for node in layer.neurons:
        node.rand_weights(dim)
      dim = layer.nb_neurons

  def create_mini_batches(self, x, y, batch_size): 
    mini_batches = [] 
    data = np.hstack((x, y)) 
    np.random.shuffle(data) 
    n_minibatches = data.shape[0] // batch_size 
    i = 0
  
    for i in range(n_minibatches + 1): 
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
        x_mini = mini_batch[:, :-1] 
        y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((x_mini, y_mini)) 
    if data.shape[0] % batch_size != 0: 
        mini_batch = data[i * batch_size:data.shape[0]] 
        x_mini = mini_batch[:, :-1] 
        y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((x_mini, y_mini)) 
    return mini_batches 