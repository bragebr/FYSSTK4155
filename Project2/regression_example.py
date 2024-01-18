from NeuralNetwork import *
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

"""
TRAINING EXAMPLE - REGRESSION ON EXPONENTIAL FUNCTION WITH TAYLOR SERIES

"""

X,Y, X_train, X_test, t_train, t_test = set_data_('LINEAR')

net = NeuralNetwork()

HIDDEN_FUNC_1, HIDDEN_GRAD_1 = LINEAR, LINEAR_PRIME

HIDDEN_FUNC_2, HIDDEN_GRAD_2 = LINEAR, LINEAR_PRIME

OUTPUT_FUNC, OUTPUT_GRAD = LINEAR, LINEAR_PRIME

net.add(ConnectLayers(X_train.shape[1],X_train.shape[1]))
net.add(ActivationLayer(HIDDEN_FUNC_1, HIDDEN_GRAD_1))
net.add(ConnectLayers(X_train.shape[1],2 * X_train.shape[1]))
net.add(ActivationLayer(HIDDEN_FUNC_2, HIDDEN_GRAD_2))
net.add(ConnectLayers(2 * X_train.shape[1],X_train.shape[1]))
net.add(ActivationLayer(OUTPUT_FUNC, OUTPUT_GRAD))
net.add(ConnectLayers(X_train.shape[1],1))

net.set_cost_function(MAE,MAE_PRIME)

net.train(X_train,t_train,10000, 0.01)

prediction = net.predict(X_test)

print('Test error for linear regression is', MAE(t_test, prediction))

pred_on_full_data_set = net.predict(X)

err = (1/(2 * len(Y))) * np.sum((Y - pred_on_full_data_set) ** 2) # MSE

print(f'MAE for prediction on full data set: {err}')

plt.title(f'Taylor Series Approximation of an Exponential Function''\n'
'Two Hidden Layers | Linear Activations | 10e5 Epochs')
plt.plot(np.linspace(-1.5,1.5,X.shape[0]), Y, 'ro', label = 'Target',
 alpha = 0.2)
plt.plot(np.linspace(-1.5,1.5,X.shape[0]), pred_on_full_data_set,
label = 'Fit')
plt.savefig('taylorapprox.png')
plt.legend(); plt.xlabel('x'); plt.ylabel('y')

plt.show()
