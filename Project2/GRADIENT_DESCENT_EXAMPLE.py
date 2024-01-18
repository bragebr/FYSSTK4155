from math import exp, sqrt
from random import random, seed
import jax.numpy as np
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from jax import grad, jit
from numpy import random as rnd
import seaborn as sns
import pandas as pd
from tabulate import tabulate
import time

import mylib2 as mlb # AUTHORS LIBRARY OF GD FUNCTIONS

# FOR PLOT STYLES
sns.set_theme()

# COST FUNCTION DEFINITIONS - OLS AND RIDGE TYPE
# SGD HAS ITS OWN DEFINITIONS AS IT NEEDS MORE INPUTS

def COST_FUNC(beta):

    return (1/2) * ((y-X @ beta)**2).mean()

def RIDGE_FUNC(beta):

    return (((y - X @ beta)**2) + (const * beta.T @ beta)).mean()

def SGD_COST(X,y,beta):

    return (1/2) * ((y-X @ beta)**2).mean()

def SGD_RIDGE(X,y,beta):

    return (((y - X @ beta)**2) + (const * beta.T @ beta)).mean()

n = 100 # NO. data points
rnd.seed(0)

# create virtual data
x = np.linspace(-1.5, 1.5, n).reshape(-1, 1)

y = np.exp(-x**2) + 0.5 * rnd.normal(0, 0.1, x.shape) + 1.5 * np.exp(-(x-2)**2)


maxd = 5 # maximum no. of degrees in the design matrix

const = 1e-4 # RIDGE CONSTANT

X = np.column_stack([x**i for i in range(maxd)]) # FEATURE MATRIX

XT_X = np.linalg.pinv(X.T @ X)

B = XT_X @ X.T @ y # COMPUTED COEFFICIENTS WITH MATRIX INVERSION
                   # FOR COMPARISON

H = XT_X * (2.0/n) # HESSIAN
eigval, eigvec = np.linalg.eig(H)
# STARTING GUESS CAN BE MADE FROM 1/np.max(eigval)


theta_out = rnd.rand(maxd,1) # INITIAL GUESS

# MAKE MODELS FOR EACH OPTIMIZER

OPTIMIZERS = {1:'GDM', 2:'AdaGrad', 3:'RMSprop', 4:'ADAM'}
models = {}
losses = {}
models_sgd = {}
losses_sgd = {}

for k in range(1,len(OPTIMIZERS)+1):
    theta, step = mlb.GRADIENT_DESCENT(COST_FUNC, OPTIMIZERS[k],
    theta_out, 0.01, 100)
    theta_sgd = mlb.SGD(SGD_COST, OPTIMIZERS[k],
    theta_out, X, y, 0.05, 100)

    cost = COST_FUNC(theta)
    cost_sgd = SGD_COST(X,y,theta_sgd)

    models.update({OPTIMIZERS[k] : X @ theta})
    models_sgd.update({OPTIMIZERS[k] : X @ theta_sgd})

    losses.update({OPTIMIZERS[k] : cost})
    losses_sgd.update({OPTIMIZERS[k] : cost_sgd})

header = ['Optimizer', 'Loss']
loss_data = tabulate(zip(losses.keys(), losses.values()), header)
loss_sgd_data = tabulate(zip(losses_sgd.keys(), losses_sgd.values()), header)
print("Non-stochastic optimizers")
print(loss_data)
print("Stochastic optimizers")
print(loss_sgd_data)



# PLOT
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.set_title("Optimizers Under Plain GD")
ax2.set_title("Optimizers Under SGD")
for model in models:
    ax1.plot(x, models[model], linestyle = '--', label = model)
    ax2.plot(x, models_sgd[model], linestyle = '--', label = model)

ax1.set_ylabel('y'); ax1.set_xlabel('x'); ax2.set_xlabel('x')
ax1.plot(x, y, 'ro', label = 'Target', alpha = 0.1); ax1.grid()
ax2.plot(x, y, 'ro', label = 'Target', alpha = 0.1); ax2.grid()

plt.legend(); plt.savefig("OPTIMIZERS.png")
plt.show()

# Trial to see loss variation as a function of no. of epochs in SGD
epochs = [int(i) for i in range(0,150,5)]

losses_epochs = {}
losses_optimizer = {}

"""
for j in range(1, len(OPTIMIZERS)+1):
    for epoch in epochs:
        theta = mlb.SGD(SGD_COST, OPTIMIZERS[j], theta_out, X,y, 0.01, epoch)

        loss = SGD_COST(X,y,theta)

        losses_epochs.update({epoch : np.log10(loss)})
    losses_optimizer.update({OPTIMIZERS[j] : dict(losses_epochs)})

plt.title("Model Loss as a Function of NO. Epochs")
for model in losses_optimizer:
    plt.plot(losses_optimizer[model].keys(), losses_optimizer[model].values(),
    label = model, linestyle = ':')

plt.ylabel(r"Loss ($\log_{10}$)"); plt.xlabel("Epochs")
plt.legend()
plt.savefig("LOSSEPOCHS.png")
plt.show()
"""

"""
PASS SGD METHOD THROUGH A GRID OF LEARNING RATES AND LAMBDA VALUES
COMPUTE THE LOSS WITH COST FUNCTION
SEARCH MIN VALUE AND CORRESPONDING INDICES IN MATRIX ((LEARNING RATE, LAMBDAS))
PERFORM NEW COMPUTATION WITH OPTIMAL PARAMETERS
"""

lrs = numpy.logspace(-6,1,15) # LEARNING RATES
lmds = numpy.logspace(-10,1,15) # LAMBDAS (RIDGE)

mylist = numpy.zeros((len(lrs), len(lmds))) # LR/LMD MESH

start_timer = time.perf_counter()
for lr in range(len(lrs)):
    print(lr)
    for lmd in range(len(lmds)):

        const = lmds[lmd]

        #theta = mlb.GRADIENT_DESCENT(RIDGE_FUNC, 'ADAM', theta_out, lr, 1000)
        theta_sgd = mlb.SGD(SGD_RIDGE, 'ADAM', theta_out, X,y, lrs[lr], 100)

        #loss = RIDGE_FUNC(theta)
        loss_sgd = SGD_RIDGE(X,y,theta_sgd)

        #loss_lmd.update({const : loss})
        mylist[lr][lmd] = loss_sgd
end_timer = time.perf_counter()

print("Grid made in", end_timer - start_timer, "seconds.")

# FIND INDICES OF MIN VALUE - RETURNS (ROW IDX, COL IDX)
minidc = numpy.unravel_index(mylist.argmin(), mylist.shape)
final_timer = time.perf_counter()
print("Min from index", mylist[minidc[0]][minidc[1]])
print("Learning rate:", lrs[minidc[0]])
print("Lambda:", lmds[minidc[1]])
print("Values found in", final_timer - end_timer, "seconds")

sns.heatmap(mylist, annot=True, cmap = 'summer')
plt.show()

const = lmds[minidc[1]]
trial_theta = mlb.SGD(SGD_RIDGE, 'ADAM', theta_out, X,y, lrs[minidc[0]], 100)

cost = (1/(2*len(y))) * np.sum((y - X @ trial_theta) ** 2)

print(f'Numerical loss (MSE) under ADAM regression with optimal parameters: {cost}')

plt.title('Polynomial Fit of Exponential Function Under ADAM Optimization')
plt.xlabel('x'); plt.ylabel('y')
plt.plot(x, X @ trial_theta, linestyle = ':', label = 'Approximation')
plt.plot(x, y, 'ro', label = 'Data',alpha = 0.2,
)
plt.legend(); plt.savefig('ADAMopt.png')
#plt.plot(x, true_func, label = 'True Function')
plt.show()
