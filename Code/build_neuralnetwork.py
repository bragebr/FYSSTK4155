# Import required libraries2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import time
import warnings
warnings.simplefilter('ignore')

from load_datasets import ricci, linear

# Import necessary modules
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from math import sqrt

# Keras specific
from tensorflow import keras
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.initializers import GlorotUniform


training_set, test_set = ricci()
X_train, y_train = training_set
X_test,  y_test  = test_set


#==============================================================================
# Function: _compile
# _compile builds a deep neural network via Tensorflow Keras by taking
# array inputs such as number of nodes in layers (units), booleans and 'choice'
# arrays such as what activation function to use (activation)
#==============================================================================



def _compile(units,
             units1,
             units2,
             activation,
             activation1,
             activation2,
             dropout,
             dropout1,
             dropout2,
             third_layer,
             learning_rate):

    model = Sequential() # initialize sequential model
    model.add(Dense(units=units,
                    activation=activation)) # input layer with varying NO. nodes
                                            # and activation functions

    if dropout: # dropout layer for better control of global weights
        model.add(Dropout(rate=0.15))

    model.add(Dense(units=units1, activation=activation1)) # first hidden layer
                                                           # with varying NO.
                                                           # nodes and activation
                                                           # functions
    if dropout1:
        model.add(Dropout(rate=0.15))

    if third_layer: # third layer option
        model.add(Dense(units = units2, activation=activation2))

        if dropout2:
            model.add(Dropout(rate=0.25))

    model.add(Dense(1)) # output layer

    model.compile(
    optimizer  = keras.optimizers.legacy.RMSprop(learning_rate=learning_rate),
    loss       = 'mean_squared_error',
    metrics    = [keras.metrics.MeanSquaredError()]
    )

    return model

#==============================================================================
# Function: _build_model
# Wraps the regression model made with function _compile with input arrays
# units: choice of NO. nodes. Passed to all layers in the network
# activation: choice of activation function
# dropout: boolean - add dropout layer or not
# third_layer: boolean - add third layer or not
# learning_rate: choice of learning rate passed to optmizer
#==============================================================================

#==============================================================================
"""
UPDATE NOTE 18/12/23:

This code no longer uses the function _compile. Wrapping the regressor as first
intented showed to be problematic with the implementation of
'keras_tuner.Hyperband()' for hyperparameter tuning. At its current state, this
code has the function _compile hard-coded into the function _build_model.

P.S. IF RUNNING THIS PROGRAM:
The code will print a warning message for each iteration of hyperband. This is
because the optimizer keras.optimizers.RMSprop functions slower than optimally
on Mac M1/M2 chips (like on the authors computer). It suggests changing to
keras.optimizers.legacy.RMSprop - do not change this as it interferes with
keras_tuner.Hyperband() and throws errors.
"""
#==============================================================================


def build_model(hp):

    units = hp.Int('units',
                    min_value = X_train.shape[1],
                    max_value = 2 * X_train.shape[1],
                    step = 10)
    units1 = hp.Int('units1',
                    min_value = 0.5 * (X_train.shape[1]+1),
                    max_value = 2 * X_train.shape[1],
                    step = 20)
    units2 = hp.Int('units2',
                    min_value = 1,
                    max_value = X_train.shape[1],
                    step = 10)

    activation = hp.Choice('activation',
                           ['relu', 'elu', 'tanh'])
    activation1 = hp.Choice('activation1',
                            ['relu', 'elu', 'tanh'])
    activation2 = hp.Choice('activation2',
                            ['relu', 'elu', 'tanh'])

    dropout  = hp.Boolean('dropout')
    dropout1 = hp.Boolean('dropout1')
    dropout2 = hp.Boolean('dropout2')
    third_layer = hp.Boolean('third_layer')


    learning_rates = hp.Float('lr',
                              min_value = 1e-4,
                              max_value = 1e-1,
                              sampling = 'log')

    model = Sequential() # initialize sequential model
    model.add(Dense(units=units,
                    activation=activation)) # input layer with varying NO. nodes
                                            # and activation functions

    if dropout: # dropout layer for better control of global weights
        model.add(Dropout(rate=0.15))

    model.add(Dense(units=units1, activation=activation1)) # first hidden layer
                                                           # with varying NO.
                                                           # nodes and activation
                                                           # functions
    if dropout1:
        model.add(Dropout(rate=0.15))

    if third_layer: # third layer option
        model.add(Dense(units = units2, activation=activation2))

        if dropout2:
            model.add(Dropout(rate=0.25))

    model.add(Dense(1)) # output layer

    model.compile(
    optimizer  = keras.optimizers.RMSprop(learning_rate=learning_rates),
    loss       = 'mean_absolute_error',
    metrics    = [keras.metrics.MeanAbsoluteError()]
    )

    """
    model = _compile(units        = units,
                     units1       = units1,
                     units2       = units2,

                     activation   = activation,
                     activation1  = activation1,
                     activation2  = activation2,

                     dropout      = dropout,
                     dropout1     = dropout1,
                     dropout2     = dropout2,

                     third_layer   = third_layer,
                     learning_rate = learning_rate)
    """

    return model

#==============================================================================
# Create tuner object using the built regressor to find the optimal
# parameter configuration through a given number of trials.

"""
Hyperband() works by choosing a handful of the fittest candidates from one
trial round to go forth into a new trial round and so on. 
"""
#==============================================================================

tuner = kt.Hyperband(build_model,
                     objective  = kt.Objective('val_mean_absolute_error',
                                                   direction = 'min'),
                     max_epochs = 200,
                     factor = 20,
                     overwrite = True,
                     directory = './',
                     project_name = 'model_tuning'
)

tuner.search_space_summary()

start_search = time.perf_counter()

tuner.search(X_train,
             y_train,
             epochs=150,
             validation_data=(X_test,y_test),
             callbacks=[keras.callbacks.EarlyStopping(patience=15)],
             verbose=0)

finish_search = time.perf_counter()
print(f'Search completed in {finish_search - start_search} sec.')

tuner.results_summary()

#==============================================================================
# Create a tuned regressor with build_model function. This is the final model
# used for training and prediction. Can also be used for refinement tuning.
#==============================================================================

tuned_regressor = tuner.hypermodel.build(
            tuner.get_best_hyperparameters(num_trials=1)[0])

tuned_regressor.fit(X_train,
                    y_train,
                    epochs=200,
                    batch_size = 25,
                    validation_split = 0.1,
                    callbacks=[keras.callbacks.EarlyStopping(patience=25)])

tuned_regressor.save('./tuned_regressor.keras')
fin = time.perf_counter()

print('REGRESSION REPORT')
print(f'Hyperparameter search and regression performed in {fin - start_search} s')
print(f'Tuned regressor model and summary saved to working directory.')
