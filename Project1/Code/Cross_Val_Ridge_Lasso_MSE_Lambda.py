import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold



def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)		# Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:, q+k] = (x**(i-k))*(y**k)

    return X


np.random.seed(125)

d = 8  # polynomial degree
N = 100   # number of datapoints
k1 = 10    # min k-fold in Cross-Validation
k2 = 20    # max k-fold in Cross-Validation

nlambdas = 100
Lambda = np.logspace(-8, 8, nlambdas)

# Create arrays for data storage
MSE_Predict_Ridge = np.zeros(nlambdas)
MSE_Train_Ridge = np.zeros(nlambdas)
MSE_Predict_Lasso = np.zeros(nlambdas)
MSE_Train_Lasso = np.zeros(nlambdas)


# Making meshgrid of datapoints and compute Franke's function
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
noise = 0.1*np.random.normal(0, 1, x.shape)
z = FrankeFunction(x, y) + noise

# Design matrix
xy = create_X(x, y, d)

# Goes throug k-values from 5 to 10
for k in range(k1, k2):
    # Initialize a KFold instance
    kfold = KFold(k)

    # Perform the cross-validation to estimate MSE

    for i in range(nlambdas):
        print(i)

        RegRidge = Ridge(Lambda[i])
        RegLasso = Lasso(Lambda[i])

        MSEPredictRidge = np.zeros((nlambdas, k))
        MSETrainRidge   = np.zeros((nlambdas, k))
        MSEPredictLasso = np.zeros((nlambdas, k))
        MSETrainLasso   = np.zeros((nlambdas, k))

        j = 0
        # Loop throug all k-folds
        for train_inds, test_inds in kfold.split(xy):
            xy_train = xy[train_inds]
            z_train  = z[train_inds]

            xy_test = xy[test_inds]
            z_test  = z[test_inds]

            RegRidge.fit(xy_train, z_train)
            zPredictRidge = RegRidge.predict(xy_test)
            zTrainRidge = RegRidge.predict(xy_train)
            MSEPredictRidge[i, j] = np.mean((z_test - zPredictRidge)**2)
            MSETrainRidge[i, j] = np.mean((z_train - zTrainRidge)**2)


            RegLasso.fit(xy_train, z_train)
            zPredictLasso = RegLasso.predict(xy_test)
            zTrainLasso = RegLasso.predict(xy_train)
            MSEPredictLasso[i, j] = np.mean((z_test - zPredictLasso)**2)
            MSETrainLasso[i, j] = np.mean((z_train - zTrainLasso)**2)

            j += 1

        # taking the mean of all the datsets
        MSE_Predict_Ridge[i] = np.mean(MSEPredictRidge)
        MSE_Train_Ridge[i]   = np.mean(MSETrainRidge)
        MSE_Predict_Lasso[i] = np.mean(MSEPredictLasso)
        MSE_Train_Lasso[i]   = np.mean(MSETrainLasso)



    # Plot MSE for train and test data
    plt.figure()
    plt.plot(Lambda, MSE_Predict_Ridge, 'r', label='Ridge Test')
    plt.plot(Lambda, MSE_Train_Ridge, 'k--', label='Ridge Train')
    plt.plot(Lambda, MSE_Predict_Lasso, 'b', label='Lasso Test')
    plt.plot(Lambda, MSE_Train_Lasso, 'y--', label='Lasso Train')
    plt.xscale('log')
    plt.yscale('log')
    ax = plt.gca()
    ax.set_ylim([10**-4, 10**-2])
    plt.grid()
    plt.xlabel('Lambdas')
    plt.ylabel('MSE')
    plt.title('Cross-Fit - Ridge and Lasso - Franke, N:' + str(N)
              + '  k-fold:' + str(k)
              + '  d:' + str(d))
    plt.legend()
    plt.show()
