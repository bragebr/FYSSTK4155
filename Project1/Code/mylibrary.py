import sklearn.linear_model as skl
import numpy as np
import pandas as pd
import os
import io
from sklearn.utils import resample
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings("ignore")
import time
import csv
from tabulate import tabulate


# DETERMINE MEAN SQUARED ERROR BETWEEN ORIGINAL DATA AND FITTED MODEL
def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model) ** 2) / n
# DETERMINE R2 SCORE OF FITTED MODEL
def R2(y_data, y_model):
    return 1 - np.sum((y_data-y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
# DEFINITON OF FRANKE FUNCTION
def FrankeFunction(x,y,n):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    # OPTIONALLY ADD NOISE TO DATA
    noise = 0.01*np.random.normal(0,1,n)
    return term1 + term2 + term3 + term4 + noise
# CREATE DATA POINTS ON UNIFORMLY DISTRIBUTED RANDOM MESH
def xy_data(nopoints):
    x = np.sort(np.random.uniform(0, 1, nopoints))
    y = np.sort(np.random.uniform(0, 1, nopoints))
    return x,y
# CREATE DESIGN MATRIX WITH BIVARIATE POLYNOMIALS OF DEGREE d IN X AND Y
def create_X(x_data, y_data, n ):
    if len(x_data.shape) > 1:
        x = np.ravel(x_data)
        y = np.ravel(y_data)

    N = len(x_data)
    l = int((n+1)*(n+2)/2)		# Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x_data**(i-k))*(y_data**k)

    return X
# SPLIT DESIGN MATRIX IN TRAIN/TEST AND SCALE DATA
def SPLITSCALE(x_data,y_data,z_data,degree):
    ### FUNCTION create_X MAKES FEATURE MATRIX X WITH POLYNOMIALS IN X AND Y ###


    X = create_X(x_data, y_data, degree)

    # SPLIT DATA
    X_train, X_test, z_train, z_test = train_test_split(X,z_data,test_size=0.2)


    # SCALE
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, z_train, z_test
# PERFORM ORDINARY LEAST SQUARES REGRESSION
def PERFORM_OLS(degree, x_dat, y_dat, z_dat):

    X_r, X_e, z_r, z_e = SPLITSCALE(x_dat,y_dat,z_dat,degree)

    # COMPUTE PARAMETERS BETA
    betaOLS = np.linalg.inv(X_r.T @ X_r) @ X_r.T @ z_r

    # COMPUTE FIT AND PREDICTOR
    zfit = X_r @ betaOLS
    zpredictOLS = X_e @ betaOLS

    return X_r, X_e, z_r, z_e, zfit, zpredictOLS
# PERFORM RIDGE REGRESSION
def PERFORM_RIDGE(degree,x_dat,y_dat,z_dat,lmda):

    X_r, X_e, z_r, z_e = SPLITSCALE(x_dat,y_dat,z_dat,degree)
    I_mat = np.identity(np.size(X_r[0,:]))

    # COMPUTE PARAMETERS BETA
    betaRidge = np.linalg.inv(X_r.T @ X_r + lmda * I_mat) @ X_r.T @ z_r

    # COMPUTE PARAMETERS BETA
    RidgeFit = X_r @ betaRidge
    RidgePredict = X_e @ betaRidge

    return z_r, z_e, RidgeFit, RidgePredict
# PERFORM LASSO REGRESSION
def PERFORM_LASSO(degree, x_dat,y_dat,z_dat,lmda):

    X_r, X_e, z_r, z_e = SPLITSCALE(x_dat,y_dat,z_dat,degree)

    LASSOMODEL = skl.Lasso(lmda, max_iter=100)
    LASSOFit = LASSOMODEL.fit(X_r, z_r)
    LASSOPredict = LASSOMODEL.predict(X_e)

    return z_r, z_e, skl.Lasso(lmda).fit(X_r,z_r),skl.Lasso(lmda).fit(X_r,z_r).predict(X_e)


# Find optimal fitting degree and hyperparameter lambda
# DATA1 MUST BE MSE DATA, DATA2 MUST BE R2 SCORES
def FIND_BEST_PARAMS(boolean, data1, data2, lambdas):
    MSE_RIDGE_DEGREE = {}
    R2_RIDGE_DEGREE = {}
    MSE_LASSO_DEGREE = {}
    R2_LASSO_DEGREE = {}
    LAMBDA_RIDGE_DEGREE = {}
    LAMBDA_LASSO_DEGREE = {}
    degree = len(list(data1.keys()))

    for i in range(1,degree+1):
        # INDEX THE LOWEST MSE PER FITTING DEGREE
        # STORE LOWEST MSE AND CORRESPONDING LAMBDA VALUES
        # FOR EACH DEGREE FOR PLOTTING AND TABULATING
        minindex = [data1[i].index(j) for j in data1[i] if j == min(data1[i])]
        maxindex = [data2[i].index(j) for j in data2[i] if j == max(data2[i])]
        if boolean == "RIDGE":

            MSE_RIDGE_DEGREE[i] = data1[i][minindex[0]]
            R2_RIDGE_DEGREE[i] = data2[i][maxindex[0]]
            LAMBDA_RIDGE_DEGREE[i] = lambdas[minindex[0]]


        elif boolean == "LASSO":

            MSE_LASSO_DEGREE[i] = data1[i][minindex[0]]
            R2_LASSO_DEGREE[i] = data2[i][maxindex[0]]
            LAMBDA_LASSO_DEGREE[i] = lambdas[minindex[0]]


    # DETERMINE THE OPTIMAL FITTING DEGREE AND LAMBDA
    if boolean == "RIDGE":

        optdegreeRidge = min(MSE_RIDGE_DEGREE.items(), key=lambda x: x[1])[0]
        optlambdaRidge = LAMBDA_RIDGE_DEGREE[optdegreeRidge]

    elif boolean == "LASSO":

        optdegreeLASSO = min(MSE_LASSO_DEGREE.items(), key=lambda x: x[1])[0]
        optlambdaLASSO = LAMBDA_LASSO_DEGREE[optdegreeLASSO]

    if boolean == "RIDGE":

        return [
        list(MSE_RIDGE_DEGREE.keys()),
        list(MSE_RIDGE_DEGREE.values()),
        list(R2_RIDGE_DEGREE.values()),
        list(LAMBDA_RIDGE_DEGREE.values()),
        optdegreeRidge,
        optlambdaRidge]

    elif boolean == "LASSO":

        return [
        list(MSE_LASSO_DEGREE.keys()),
        list(MSE_LASSO_DEGREE.values()),
        list(R2_LASSO_DEGREE.values()),
        list(LAMBDA_LASSO_DEGREE.values()),
        optdegreeLASSO,
        optlambdaLASSO]




def BOOTSTRAP(num_bootstraps, X_train, X_test, z_train, z_test):

    z_pred = np.empty((z_test.ravel().shape[0], num_bootstraps))
    MSE_resample = np.zeros(num_bootstraps)

    for i in range(num_bootstraps):
        X_b, z_b = resample(X_train, z_train)
        beta_b = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ z_b
        z_pred[:,i] = (X_test @ beta_b).ravel()
        MSE_resample[i] = np.mean((z_test.ravel() - z_pred[:,i])**2)

    return z_pred, MSE_resample


# WRITE DATA TO CSV FILE FOR TABULATING / PLOTTING
def Write(filename, data):
    FILEPATH = "Project1/Data/"
    if len(data) == 6:
        fixed_values = [data[-2],data[-1]]
        data = zip(data[0],data[1],data[2],data[3])
        header1 = ["Optimal Degree", "Optimal Hyperparameter"]
        header2 = ["Degree", "MSE", "R2 Score", "Hyperparameter"]
        with open(FILEPATH+filename + '.csv', 'w') as f:
           writer = csv.writer(f, delimiter='\t')
           writer.writerow(header1)
           writer.writerow(fixed_values)
           writer.writerow(header2)
           writer.writerows(data)
    elif len(data) < 4:
        data = zip(data[0],data[1],data[2])
        header = ["Error", "Bias", "Variance"]
        with open(FILEPATH+filename + '.csv', 'w') as f:
           writer = csv.writer(f, delimiter='\t')
           writer.writerow(header)
           writer.writerows(data)
    else:
        first_header = ["Optimal Fitting Degree", "Corr. MSE", "Corr. R2", "Empty"]
        fixed_values = [data[-3],data[-2],data[-1]]
        data = zip(data[0],data[1],data[2],data[3])
        header = ["Degree", "MSE (Training)", "MSE (Testing)", "R2 Score"]
        with open(FILEPATH+filename + '.csv', 'w') as f:
            writer = csv.writer(f,delimiter='\t')
            writer.writerow(first_header)
            writer.writerow(fixed_values)
            writer.writerow(header)
            writer.writerows(data)
    return

# READ SPECIFIED FILE ACCORDING TO HEADER LENGTHS
def Read(filename):

    myfile = open(filename, 'r')
    header = myfile.readline()
    polydegree = []
    MSE = []
    MSE2 = []
    R2 = []
    hyperparameter = []
    error = []
    bias = []
    var = []
    if len(header.split('\t')) == 2:
        #first_header = myfile.readline()
        fixed_values = myfile.readline().split('\t')
        fixed_values = [float(val) for val in fixed_values]
        second_header = myfile.readline()

        lines = myfile.readlines()

        for line in lines:
            row = line.split()
            polydegree.append(int(row[0]))
            MSE.append(float(row[1]))
            R2.append(float(row[2]))
            hyperparameter.append(float(row[3]))
        return polydegree, MSE, R2, hyperparameter, fixed_values

    elif len(header.split('\t')) == 3:
        #header = myfile.readline()

        lines = myfile.readlines()
        for line in lines:
            row = line.split()
            error.append(float(row[0]))
            bias.append(float(row[1]))
            var.append(float(row[2]))
        return error, bias, var
    else:
        fixed_values = myfile.readline().split('\t')
        fixed_values = [float(val) for val in fixed_values]
        second_header = myfile.readline()

        lines = myfile.readlines()
        for line in lines:
            row = line.split()
            polydegree.append(int(row[0]))
            MSE.append(float(row[1]))
            MSE2.append(float(row[2]))
            R2.append(float(row[3]))
        return polydegree, MSE, MSE2, R2, fixed_values
