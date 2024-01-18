from NeuralNetwork import *
import numpy as np

np.random.seed(0)

"""
TRAINING EXAMPLE - THE XOR GATE
THE XOR GATE RETURNS 0 FOR A BINARY INPUT OF TWO EQUAL VALUES AND
1 FOR A BINARY INPUT WHERE THE TWO INPUTS ARE DIFFERENT
THIS IS A NON LINEAR MAPPING SOLVED EASILY BY A DEEP NETWORK WITH
NON LINEAR ACTIVATION FUNCTIONS IN THE HIDDEN LAYERS
"""

X,Y_XOR, Y_AND = set_data_('GATES')

net = NeuralNetwork()

HIDDEN_FUNC_1, HIDDEN_GRAD_1 = RELU, RELU_PRIME

HIDDEN_FUNC_2, HIDDEN_GRAD_2 = RELU, RELU_PRIME

OUTPUT_FUNC, OUTPUT_GRAD = NTANH, NTANH_PRIME

net.add(ConnectLayers(X.shape[1],2))
net.add(ActivationLayer(HIDDEN_FUNC_1, HIDDEN_GRAD_1))
net.add(ConnectLayers(2,2))
net.add(ActivationLayer(HIDDEN_FUNC_2, HIDDEN_GRAD_2))
net.add(ConnectLayers(2,2))
net.add(ActivationLayer(OUTPUT_FUNC, OUTPUT_GRAD))
net.add(ConnectLayers(2,1))

net.set_cost_function(MAE,MAE_PRIME)

net.train(X,Y_XOR,10000, 0.01)

prediction = net.predict(X, type = 'Classification')

print(prediction)
