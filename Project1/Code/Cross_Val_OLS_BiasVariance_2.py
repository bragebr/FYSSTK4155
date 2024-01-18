import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4


def create_X(x, y, n ):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X


np.random.seed(125)

d = 8   # polynomial degree
N = 25   # number of datapoints
k1 = 10    # min k-fold in Cross-Validation
k2 = 11    # max k-fold in Cross-Validation


# Create arrays for data storage
SK_error = np.zeros(d)
error = np.zeros(d)
error_train = np.zeros(d)
bias = np.zeros(d)
variance = np.zeros(d)
polydegree = np.zeros(d)

# Making meshgrid of datapoints and compute Franke's function
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
noise = 0.1*np.random.normal(0,1,x.shape)
z = FrankeFunction(x, y) + noise

# Goes throug k-values from 5 to 10
for k in range(k1,k2):
    # Initialize a KFold instance
    kfold = KFold(k) 


    # Perform the cross-validation to estimate MSE

    # Loop through all degrees
    for degree in range(d):
        print(degree)
        #Design matrix
        xy = create_X(x, y, degree) 
        OLS = LinearRegression(fit_intercept=False)
        
        
        MSE_test = np.zeros(k)
        MSE_train = np.zeros(k)
        bi = np.zeros(k)
        var = np.zeros(k)
        j = 0
        # Loop throug all k-folds
        for train_inds, test_inds in kfold.split(xy): 
            xy_train = xy[train_inds]
            z_train = z[train_inds]
            
            xy_test = xy[test_inds]
            z_test = z[test_inds]
            
            model = OLS.fit(xy_train,z_train)
            z_pred = model.predict(xy_test)
            z_fit = model.predict(xy_train)
            MSE_train[j] = np.mean((z_train - z_fit)**2)
            MSE_test[j] = np.mean((z_test - z_pred)**2)
            bi[j] =  np.mean((z_test - np.mean(z_pred))**2)
            var[j] = np.var(z_pred)
            
            j += 1
            
        sk_mse = cross_val_score(OLS, xy, z, 
                                 scoring='neg_mean_squared_error', 
                                 cv=kfold)
        polydegree[degree] = degree
        # taking the mean of all the datasets
        SK_error[degree] = np.mean(-sk_mse)
        error[degree] = np.mean(MSE_test)
        error_train[degree] = np.mean(MSE_train)
        bias[degree] = np.mean(bi)
        variance[degree] = np.mean(var)

    
    # for scaling the y-axis. The first value of the variance is 
    # close to zero and affecte the scale og the y-axis a lot.
    variance[0] = min(error) * 0.1
    #print(SK_error/error)

#    """   
    # Plot MSE, Bias and Variance
    plt.figure()
    plt.plot(polydegree, error, 'r',label='Error')
    plt.plot(polydegree, SK_error, 'k--',label='SK_error')
    #plt.plot(polydegree, bias, 'b',label='Bias')
    #plt.plot(polydegree, variance, 'g',label='Variance')
    #plt.plot(polydegree, bias+variance, 'y--')
    plt.grid()
    plt.yscale('log')
    plt.xlabel('Polynomial degree')
    plt.ylabel('MSE')
    plt.title('Cross-Fit - OLS, Franke, N:' +str(N) 
              +'  k-fold=' +str(k)  
              +'  d:' +str(d))
    plt.legend()
    plt.show()


    """
    # Plot MSE for train and test data
    plt.figure()
    plt.plot(polydegree, error, 'r',label='Error Test')
    plt.plot(polydegree, error_train, 'k--',label='Error Train')
    plt.yscale('log')
    ax = plt.gca()
    ax.set_ylim([10**-4, 10**3])
    plt.grid()
    plt.xlabel('Polynomial degree')
    plt.ylabel('MSE')
    plt.title('Cross-Fit - OLS, Franke, N:' +str(N) 
              +'  k-fold=' +str(k)  
              +'  d:' +str(d))
    plt.legend()
    plt.show()


    """ 