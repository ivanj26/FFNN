import numpy as np

class MiniBatchGD:
  def __init__(self, layers, lr, momentum):
    self.lr = lr
    self.momentum = momentum
    self.layers = layers
  
  def fit(self, x, y, epochs, batch_size):
    for epoch in range(epochs):
      # split into mini batch
      mini_batches = self.create_mini_batches(x, y, batch_size) 
      for mini_batch in mini_batches: 
        x_mini, y_mini = mini_batch
        error_list = []
        # for every data (features and output target) in the mini batch
        for x_i, y_i in zip(x_mini, y_mini):
          # init feed forward from raw features
          output = self.layers[0].feed_forward(x_i)
          # pass output of layer to the next one
          for i in range(1, len(self.layers)):
            output = self.layers[i].feed_forward(output)
          print(output)          
          # calculate mean_squared_error
          # error = 0.5 * (output[0] - y_i) * (output[0] - y_i)
          # error_list.append(error)
          # # back propagate
          # deltas = [y_i - output[0]]
          # weights = [1]
          

        


  def predict(self, x):
    pass

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