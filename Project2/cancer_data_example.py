from NeuralNetwork import *
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


"""
TRAINING EXAMPLE - THE CANCER DATA
"""

X,Y, X_train, X_test, t_train, t_test = set_data_('CANCER_DAT')


net = NeuralNetwork()

HIDDEN_FUNC_1, HIDDEN_GRAD_1 = RELU, RELU_PRIME

HIDDEN_FUNC_2, HIDDEN_GRAD_2 = RELU, RELU_PRIME

OUTPUT_FUNC, OUTPUT_GRAD = NTANH, NTANH_PRIME

net.add(ConnectLayers(X_train.shape[1],X_train.shape[1]+5))
net.add(ActivationLayer(HIDDEN_FUNC_1, HIDDEN_GRAD_1))
net.add(ConnectLayers(X_train.shape[1]+5,X_train.shape[1]+2))
net.add(ActivationLayer(HIDDEN_FUNC_2, HIDDEN_GRAD_2))
net.add(ConnectLayers(X_train.shape[1]+2,X_train.shape[1]))
net.add(ActivationLayer(OUTPUT_FUNC, OUTPUT_GRAD))
net.add(ConnectLayers(X_train.shape[1],1))

net.set_cost_function(MAE,MAE_PRIME)

net.train(X_train,t_train,10000, 0.096)

prediction = net.predict(X_test, type = 'Classification')

accuracy = np.zeros(len(t_test))

for i in range(len(t_test)):

        if t_test[i] == prediction[i]:
            accuracy[i] = 1.0

score = np.sum(accuracy) / len(t_test) * 100

print(f'Accuracy on test set is {score} %' )

plt.figure(figsize=(10,5))
plt.title(f'Real and Predicted Labels for a Test Set of {t_test.shape[0]} Cases'
'\n' 'Two Hidden Layers | LReLU and NTANH Activations | 10e5 Epochs')
plt.scatter(np.arange(X_test.shape[0]), prediction,
color = 'b',label = 'Predicted Label')
plt.scatter(np.arange(X_test.shape[0]), t_test, color = 'red',
label = 'Real Label', alpha = 0.4)
plt.ylabel('Label'); plt.xlabel('Case')
plt.legend(loc = 'center right') ; plt.show()
