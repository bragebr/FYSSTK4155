import mylibrary as mlb # AUTHORS LIBRARY OF FUNCTION CALLS
import numpy as np

### MAIN ###

n = 100 # number of sample points (virtual data)
d = 10 # number of fitting degree trials (OLS)
maxd = 20 # number of fitting degree trials (RIDGE/LASSO)
nlam = 50 # NO. hyperparameter trials
nlamLASSO = 10
minlam = -8
maxlam = -1
lambdas = np.logspace(minlam,maxlam,nlam) # logspace mesh of hyperparameters
lambdasLASSO = np.logspace(minlam,maxlam,nlamLASSO)

nbootstraps = n # NO. of bootstrapping cycles

np.random.seed(2023)

#x,y = mlb.xy_data(n) # virtual x and y values
#xs,ys = np.meshgrid(x,y) # meshed net of x and y values
#z = mlb.FrankeFunction(xs,ys,n) # virtual data for benchmarking the code

#"""
terrain = imread('Project1/Data/SRTM_data_Norway_1.tif') # read terrain file
z = terrain

if z.shape[0] > z.shape[1]:
    N = z.shape[1]
else: N = z.shape[0]

N = 150

# DETERMINE XY SEGMENT TO STUDY BY NO. OF POINTS N
x = np.linspace(0,N,N)
y = np.linspace(1.5*N,2.5*N,N)

xx,yy = np.meshgrid(x,y)

z = z[:len(x),:len(y)]

# NORMALIZE DATA
std = np.std(z)
z = (z - np.mean(z))/std

#"""

# DICTIONARIES FOR STORING VARIABLES (OLS)
MSEOLS = {j:0 for j in range(1,d+1)}
MSEOLSPredict = {j:0 for j in range(1,d+1)}
R2OLS = {j:0 for j in range(1,d+1)}
error = {}
bias = {}
var = {}

# COMMENT IN BOOTSTRAP FUNCTION AND ERROR,BIAS,VAR FOR BIAS - VAR STUDIES

for i in range(1,d+1):
    X_train, X_test, z_train, z_test, zfit, zpredict = mlb.PERFORM_OLS(i,x,y,z)

    MSEOLS[i] = mlb.MSE(z_train,zfit)
    MSEOLSPredict[i] = mlb.MSE(z_test,zpredict)
    R2OLS[i] = mlb.R2(z_test,zpredict)

    #z_pred, MSE_resample = mlb.BOOTSTRAP(nbootstraps,X_train,X_test,z_train,z_test)

    #error[i] = np.mean(MSE_resample)
    #bias[i] = np.mean((z_test.ravel() - np.mean(z_pred,axis=1))**2)
    #var[i] = np.mean(np.var(z_pred,axis=1))

#mlb.Write('biasvar', [list(error.values()), list(bias.values()), list(var.values())])

optdegreeOLS = min(MSEOLSPredict.items(), key=lambda x:x[1])[0]
lowestMSEOLS = MSEOLSPredict[optdegreeOLS]
bestR2 = R2OLS[optdegreeOLS]


dataOLS = [
list(MSEOLS.keys()),
list(MSEOLS.values()),
list(MSEOLSPredict.values()),
list(R2OLS.values()),
optdegreeOLS,
lowestMSEOLS,
bestR2
]

# WRITE OLS DATA TO FILE FOR PLOTTING/TABULATING
mlb.Write('OLSDat_real', dataOLS)

#"""
# DICTIONARIES FOR STORING VECTORS OF VARIABLES (RIDGE)
MSERIDGE = {j:[] for j in range(1,maxd+1)}
MSERIDGEPredict = {j:[] for j in range(1,maxd+1)}
R2RIDGE = {j:[] for j in range(1,maxd+1)}

for i in range(1,maxd+1):
    for j in lambdas:
        z_train, z_test, RidgeFit, RidgePredict = mlb.PERFORM_RIDGE(i,x,y,z,j)

        MSERIDGE[i].append(mlb.MSE(z_train, RidgeFit))
        MSERIDGEPredict[i].append(mlb.MSE(z_test,RidgePredict))
        R2RIDGE[i].append(mlb.R2(z_test,RidgePredict))

paramsRidge = mlb.FIND_BEST_PARAMS("RIDGE", MSERIDGEPredict, R2RIDGE, lambdas)

# WRITE RIDGE DATA TO FILE FOR PLOTTING/TABULATING
mlb.Write("RidgeDat_real", paramsRidge)
#"""
#"""
# DICTIONARIES FOR STORING VECTORS OF VARIABLES (LASSO)
MSELASSO = {j:[] for j in range(1,maxd+1)}
MSELASSOPredict = {j:[] for j in range(1,maxd+1)}
R2LASSO = {j:[] for j in range(1,maxd+1)}

for i in range(1,maxd+1):
    for j in lambdasLASSO:
        z_train, z_test, LASSOFit, LASSOPredict = mlb.PERFORM_LASSO(i,x,y,z,j)
        #print(LASSOFit)
        #MSELASSO[i].append(mlb.MSE(z_train, LASSOFit))
        MSELASSOPredict[i].append(mlb.MSE(z_test,LASSOPredict))
        R2LASSO[i].append(mlb.R2(z_test,LASSOPredict))

paramsLASSO = mlb.FIND_BEST_PARAMS("LASSO", MSELASSOPredict, R2LASSO, lambdasLASSO)

# WRITE LASSO DATA TO FILE FOR PLOTTING/TABULATING
mlb.Write("LASSOdat_real", paramsLASSO)
#"""
