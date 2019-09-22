from model.util.activation_fun import sigmoid
from model.node import Node
import numpy as np

inputs = [1,2,3,4,5]
len = len(inputs)

node = Node(sigmoid)
node.rand_weights(len)
print(node.calc_net(inputs))