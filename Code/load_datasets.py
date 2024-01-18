import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


# = = = = = = = = = = = = = = = = = = = = = = = = = =
# FUNCTION SCALE_SPLIT returns scaled and split
# data used for training and testing the regressor
# models
# = = = = = = = = = = = = = = = = = = = = = = = = = =


def SCALE_SPLIT(x_dat, y_dat, scale=True,split=True):
    selector = VarianceThreshold()
    scaler = StandardScaler()

    if scale == False:
        X_train,X_test,Y_train,Y_test = train_test_split(x_dat,
                                                                y_dat,
                                                                test_size=0.2,
                                                                random_state=42)

        return X_train,X_test,Y_train,Y_test


    x_scaled = scaler.fit_transform(x_dat)
    y_dat = np.log10(y_dat)
    y_scaled = scaler.fit_transform(y_dat)

    if split == False:

        return x_scaled, y_scaled

    X_train, X_test, Y_train, Y_test = train_test_split(x_scaled,
                                                        y_scaled,
                                                        test_size = 0.20,
    random_state=42)

    return X_train, X_test, Y_train, Y_test

def get_full_ricci():

    data = pd.read_pickle('Project3/Datasets/ricci_conductivity_dataset.pkl')

    data = data.dropna() # drop columns w/ missing values

    data = data.rename(columns={'conductivity' : 'target'})

    data_set = data.drop(
    ['pretty_formula', 'structure', 'composition', 'composition_oxid'],
    axis = 1)

    X = data_set.drop('target', axis=1).values
    y = data_set.target.values.reshape(-1,1)

    X,y = SCALE_SPLIT(X,y,split=False)

    return X,y




# = = = = = = = = = = = = = = = = = = = = = = = = = =
# FUNCTION linear() returns training and testing data
# for a linear function. This is used
# for benchmarking and testing of regressor codes
# = = = = = = = = = = = = = = = = = = = = = = = = = =

def get_full_linear():

    x = np.linspace(-1.5,1.5,3000).reshape(-1,1)
    noise = 0.2 * np.random.normal(0,0.2,x.shape)
    y_trial = np.exp(-x**2) + 2*np.exp(-(x-2)**2) + noise

    X_trial = np.column_stack([x ** i for i in range(5)])

    return x, X_trial, y_trial



def linear():

    x = np.linspace(-1.5,1.5,3000).reshape(-1,1)
    noise = 0.2 * np.random.normal(0,0.2,x.shape)
    y_trial = np.exp(-x**2) + 2*np.exp(-(x-2)**2) + noise

    X_trial = np.column_stack([x ** i for i in range(5)])

    X_train, X_test, y_train, y_test = SCALE_SPLIT(X_trial, y_trial,scale=False)

    training_set = (X_train,y_train) ; test_set = (X_test,y_test)

    return training_set, test_set

# = = = = = = = = = = = = = = = = = = = = = = = = = =
# FUNCTION ricci() returns training and testing data
# containing material features and computed thermal
# conductivity
# = = = = = = = = = = = = = = = = = = = = = = = = = =


def ricci():




    data = pd.read_pickle('Project3/Datasets/ricci_conductivity_dataset.pkl')

    data = data.dropna() # drop columns w/ missing values

    data = data.rename(columns={'conductivity' : 'target'})

    training_set = data.drop(
    ['pretty_formula', 'structure', 'composition', 'composition_oxid'],
    axis = 1)

    X = training_set.drop('target', axis=1).values
    y = training_set.target.values.reshape(-1,1)


    X_train, X_test, y_train, y_test = SCALE_SPLIT(X,y)

    training_set = (X_train,y_train) ; test_set = (X_test, y_test)

    return training_set, test_set
