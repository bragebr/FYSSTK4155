import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_unformatted_ricci():

    data = pd.read_pickle('./ricci_conductivity_dataset.pkl')

    data = data.dropna() # drop columns w/ missing values

    data = data.rename(columns={'conductivity' : 'target'})

    training_set = data.drop(
    ['pretty_formula', 'structure', 'composition', 'composition_oxid'],
    axis = 1)

    X = training_set.drop('target', axis=1)
    y = training_set.target

    return X, y


def SCALE_SPLIT(x_dat, y_dat):

    scaler = StandardScaler()

    x_scaled = scaler.fit_transform(x_dat)
    y_scaled = scaler.fit_transform(y_dat)

    X_train, X_test, Y_train, Y_test = train_test_split(x_scaled,y_scaled,random_state=42)

    return X_train, X_test, Y_train, Y_test

def linear():

    x = np.linspace(0,2*np.pi,1000).reshape(-1,1)

    y_trial = x * np.sin(x)

    X_trial = np.column_stack([x ** i for i in range(5)])

    X_train, X_test, y_train, y_test = SCALE_SPLIT(X_trial, y_trial)

    training_set = (X_train,y_train) ; test_set = (X_test,y_test)

    return training_set, test_set



def ricci():



    data = pd.read_pickle('./ricci_conductivity_dataset.pkl')

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
