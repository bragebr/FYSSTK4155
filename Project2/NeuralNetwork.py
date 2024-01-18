import numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad, grad
import autograd.numpy as autonp
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns

FILE = 'data.csv'

sns.set_theme() # PLOTTING THEME

#############################################################################
"""
ACTIVATION AND COST FUNCTION DEFINITIONS WITH THEIR ANALYTICAL DERIVATIVES
"""
#############################################################################

def SIGMOID(input):
    term = 1.0 / (1 + np.exp(-input))
    return term

def SIGMOID_PRIME(input):

    return SIGMOID(input) * (1 - SIGMOID(input))

def NTANH(input): # A sigmoidal tanh function with extra linear term for
                  # differentiation stability
    delta = 10e-10
    return 1.79 * np.tanh((2/3) * input) + (delta * input)

def NTANH_PRIME(input):
    delta = 10e-10
    return 1.79 * (2/3) / (np.cosh((2/3) * input) ** 2.0) + delta

def SOFT(input): # NOT USED
    exp = autonp.exp(input)
    return exp/ (autonp.sum(exp, axis=0, keepdims=True) + 10e-10)

def SOFT_PRIME(input): # NOT USED

    GRADFUNC = elementwise_grad(SOFT)

    return GRADFUNC(input)

def RELU(input): # return input where input > 0, else return 0
    return np.where(input > np.zeros(input.shape), input, np.zeros(input.shape))

def RELU_PRIME(input): # return 1 where input > 0, else return 0
    return np.where(input > np.zeros(input.shape), np.ones(input.shape), np.zeros(input.shape))

def LRELU(input): # return input where input > 0, else return scaled input
    delta = 10e-4
    return np.where(input > np.zeros(input.shape), input, delta * input)

def LRELU_PRIME(input): # return 1 where input > 0, else return small value
    delta = 10e-4
    return np.where(input > np.zeros(input.shape), 1, delta)

def LINEAR(input): # pass all input under activation (for regression)
    return input

def LINEAR_PRIME(input):
    return 1.0

def MSE(target, prediction): # Mean squared error (potentially unstable)

    return (1 / 2 * target.shape[0]) * np.sum((target - prediction) ** 2)

def MSE_PRIME(target, prediction):
    return (target - prediction) / target.shape[0]

def MAE(target, prediction): # Mean absolute error (stable)

    return (1 / target.shape[0]) * np.sum(np.abs(prediction - target))

def MAE_PRIME(target, prediction):
    diff = (target - prediction)
    delta = 10e-10
    return -(1 / target.shape[0]) * (diff) / (np.abs(diff) + delta)

def BINARYCROSSENTROPY(target, prediction): # log loss (unstable)
    delt = 10e-5
    N = -(1.0 / target.shape[0])
    t,p = target,prediction
    p += delt
    ones = np.ones(target.shape[0])
    return N * np.sum(t * np.log(p) + (ones - t) * np.log(ones - p))

def BINARYCROSSENTROPY_PRIME(target, prediction):

    return (1 / target.shape[0]) * np.abs((prediction - target))


################################################################################

"""
Definitions needed for a neural network object follows under.
The object NeuralNetwork has attributes:

add - add a layer. Can be a fully connected layer or the activation of a
fully connected layer.

set_cost_function - takes input (cost function, cost function derivative)

train - train the network object with inputs
(
input data, target data, learning rate, OPTIONAL weight regularizer (not stable),
OPTIONAL optimizer type (not fully implemented)
)

predict - make prediction on input data with inputs (input data, type). 'Type' is
'Regression' by default, will return a single forwardpass output on inputdata.
Optional type 'Classification' returns output after measuring against a threshold.
If (1-output) < 10e-2 the output is regarded as class 1, if output < 10e-2
the output is regarded as class 0.


"""

################################################################################
class SingleLayer:

    def __init__(self):

        self.input = None
        self.output = None


    def forwardpass(self, input):

        raise NotImplementedError

    def backpropagation(self, output_error, lr, lmd):

        raise NotImplementedError


class ConnectLayers(SingleLayer):

    def __init__(self, input_dim, output_dim):

        # HEURISTIC WEIGHT INITIALIZATION & NON-ZERO BIAS
        SCALER = np.sqrt(2 / input_dim)
        DELTA = 10e-10
        self.weights = np.random.randn(input_dim, output_dim) * SCALER
        self.bias = np.zeros(output_dim) + DELTA

    def forwardpass(self, data):

        self.input = data

        # Computation of X @ w + b
        self.output = np.dot(self.input, self.weights) + self.bias

        return self.output

    def backpropagation(self, output_error, lr,lmd, optimizer = 'constant'):

        input_error = np.matmul(output_error, self.weights.T)

        dW = np.matmul(self.input.T, output_error)

        dB = np.sum(output_error, axis=0)

        # OPTIMIZER OBJECT IS NOT IMPLEMENTED UPON DELIVERY OF THIS PROJECT
        # FUTURE WORK INCLUDES IMPLEMENTING FLEXIBLE CHOICE OF OPTIMIZER


        #self.optimizer = optimizer # DEFAULT IS CONSTANT GRADIENT DESCENT

        # WEIGHT REGULARIZER IS VERY UNSTABLE
        # FUTURE WORK INCLUDES STABILIZING L1 AND L2 REGULARIZERS
        #self.weights += lmd * self.weights



        self.weights -= lr * dW # CONSTANT GRADIENT DESCENT UPDATE

        self.bias -= lr * dB

        return input_error

    # REPEATING: OPTIMIZER DOES NOT WORK. WILL IMPROVE IN THE FUTURE
    def Optimizer(self, num_weights, lr, optimizer):

        self.m = [0] * num_weights
        self.v = [0] * num_weights
        self.t = 1
        self.lr = lr

        if optimizer == 'ADAM':

            def Adam(self, params, grads, beta1 = 0.9, beta2 = 0.999):

                updated_params = []

                for i, (param,grad) in enumerate(zip(params,grad)):
                    g2 = np.multiply(grad,grad) # HAMARD PRODUCT
                    self.m[i] = beta1 * self.m[i] + (1-beta1) * grad
                    self.v[i] = beta2 * self.v[i] + (1-beta2) * g2

                    m_corr = self.m[i] / (1 - beta1 ** self.t)
                    v_corr = self.v[i] / (1 - beta2 ** self.t)

                    param -= self.lr * m_corr / (np.sqrt(v_corr) + 10e-8)
                    updated_params.append(param)

                self.t += 1

                return updated_params


"""
ActivationLayer inherits from SingleLayer class to apply activation function
outputs in each layer for forward - and backward passes
"""
class ActivationLayer(SingleLayer):
    def __init__(self, activation, activation_prime):

        self.activation = activation
        self.activation_prime = activation_prime

    def forwardpass(self, data):

        self.input = data
        self.output = self.activation(self.input)

        return self.output

    def backpropagation(self, output_error, lr,lmd, optimizer):
        return self.activation_prime(self.input) * output_error


class NeuralNetwork:

    def __init__(self):

        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):

        self.layers.append(layer)

    def set_cost_function(self, loss, loss_prime):

        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, data, type = 'Regression'):

        self.type = type # DEFAULT TYPE IS REGRESSION
        self.n_correct_counts = 0

        output = data
        for layer in self.layers:
            output = layer.forwardpass(output)

        if type == 'Classification':

            for j in range(len(output)):

                if output[j] < 10e-2 : output[j] = 0
                elif 1 - output[j] < 10e-2 : output[j] = 1



        return output

    def train(self, X, Y, epochs, lr, lmd=1):

        errors = {}

        for i in range(epochs): # FOR EACH TRAINING STEP

            output = X # SEND DATA SET THROUGH EACH LAYER OF THE NETWORK

            for layer in self.layers:
                output = layer.forwardpass(output)

            # OUPUT ONLY THE VALUES AT THE OUTPUT LAYER AFTER ACTIVATION
            # COMPUTE ERROR IN OUTPUT LAYER
            error = self.loss_prime(Y, output)

            # FEED ERROR IN OUTPUT LAYER BACKWARDS THROUGH THE NET
            for layer in reversed(self.layers):
                error = layer.backpropagation(error,lr,lmd, 'constant')

            print('Error @ epoch'+str(i), self.loss(Y,output))

            errors.update({i : np.log10(self.loss(Y, output))})

        return errors


# FOR EASIER CHOICE OF DATA SET
def set_data_(type):

    if type == 'GATES':
        X = np.array([ [0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]],
                     )
        Y_XOR = np.array([[0],
                          [1],
                          [1],
                          [0]])

        Y_AND = np.array([[0],
                          [1],
                          [1],
                          [1]])



        return X, Y_XOR, Y_AND

    elif type == 'LINEAR':

        n = 200 # NO. DATA POINTS
        x = np.linspace(-1.3, 1.3, n).reshape(-1,1) # RESCALE IN TERMS OF PI
                                                    # FOR FOURIER FEATURES
        d = 5 # MAX NO. FEATURES

        # WORKING DATA SHAPE - COLUMN SHAPE

        FOURIER_FEATURES = [
        np.sin(x*i)+np.cos(x*j) for i in range(d) for j in range(d)
        ]

        TAYLOR_FEATURES = [
        x ** i for i in range(d)
        ]

        X = np.column_stack(TAYLOR_FEATURES) # CHANGE TO FOURIER_FEATURES FOR
                                             # FOURIER SERIES

        eps = 0.5 * np.random.normal(0,0.1,x.shape)
        Y = np.column_stack([np.exp(-x**2) + 3 * np.exp(-(x-2)**2) + eps])

        X_tr,X_te,Y_tr,Y_te = train_test_split(X,Y,test_size = 0.2)

        return X,Y,X_tr,X_te,Y_tr,Y_te


    elif type == 'CANCER_DAT':

        CancerDat = pd.read_csv(FILE)
        target = np.zeros(len(CancerDat['diagnosis']))

        # RELABEL DATA SET FROM B & M TO 0 & 1
        for i in range(len(CancerDat['diagnosis'])):
            if CancerDat['diagnosis'][i] == 'M' : target[i] = int(1)
            else : target[i] = int(0)

        # FEATURE MATRIX CHANGED TO 10 FIRST FEATURES OF DATA SET
        X = CancerDat.loc[:, 'radius_mean':'fractal_dimension_mean']

        X_scaled = X.copy()

        # MIN MAX SCALING OF EACH COLUMN FOR COMPUTATION STABILITY
        for col in X_scaled.columns:
            max = X_scaled[col].max()
            min = X_scaled[col].min()
            X_scaled[col] = (X_scaled[col] - min) / (max - min)

        X = X_scaled.values

        target = target.reshape(-1,1)

        X_tr, X_te, t_tr, t_te = train_test_split(X,target, test_size = 0.2)
        return X,target,X_tr, X_te, t_tr, t_te
