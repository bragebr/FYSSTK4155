#==============================================================
"""
Hyperparameter space tuning implemented for an xgboost regressor
(XGBRegressor). Short the choices made:

DART : the boosting method chosen is DART due to its stability
over 'boosting rounds' or larger number of estimators (trees).
As the search potentially leads to large numbers of estimators
DART provides security in stability for trials like this

MAE : mean absolute error, chosen as the evaluation metric for
both this regressor and the neural network for the reason that
the data set has outliers which through MSE (mean squared error)
would dramatically increase this metric value unfairly in a pool
of testing candidates. MAE is softer on outliers, not to mention
more intuitive.

RandomizedSearchCV : this function from sklean randomly chooses
a configuration preset and tries the wrapped regressor object
with these parameters. Here, this function is chosen over the
seemingly more conventional GridSearchCV for the sake of
efficiency
"""
#==============================================================


import xgboost as xgb
import numpy as np
import time
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, RandomizedSearchCV

from load_datasets import ricci, linear


training_set, testing_set = ricci()

X_train, y_train = training_set ; X_test, y_test = testing_set

regressor = xgb.XGBRegressor(booster = 'dart',
                             objective = 'reg:squarederror',
                             eval_metric = 'mae')

hyperspace = {
    'n_estimators':[1,5,10], # 3
    'min_child_weight':[50,100,500], # 3
    'gamma':[i/10.0 for i in range(1,6)], # 5
    'reg_alpha' : [0,0.0001,0.001,0.1], # 4
    'reg_lambda' : [0,0.00001,10,40], # 4
    'subsample':[i/10.0 for i in range(6,11)], # 4
    'colsample_bytree':[i/10.0 for i in range(6,11)], # 4
    'max_depth': [2,3,4,6,7], # 5
    'eta': np.logspace(-5,1,10), # 10
    'skip_drop' : [0.5,0.8,1.], # 3
    'rate_drop' : [0.5,0.6,0.7] # 3
}

print(hyperspace['gamma'])

CVFolds = KFold(n_splits = 5)
start = time.perf_counter()
grid = RandomizedSearchCV(regressor,
                    hyperspace,
                    scoring = 'neg_root_mean_squared_error',
                    n_iter = 200,
                    n_jobs = -1,
                    cv = CVFolds,
                    verbose = 3).fit(X_train,y_train)

params = grid.best_params_
print(params)


hypermodel = xgb.XGBRegressor(booster = 'dart',
                             objective = 'reg:squarederror',
                             eval_metric = 'mae',
                             n_estimators = params['n_estimators'],
                             min_child_weight = params['min_child_weight'],
                             gamma = params['gamma'],
                             reg_alpha = params['reg_alpha'],
                             reg_lambda = params['reg_lambda'],
                             subsample = params['subsample'],
                             colsample_bytree = params['colsample_bytree'],
                             max_depth = params['max_depth'],
                             eta = params['eta'],
                             skip_drop = params['skip_drop'],
                             rate_drop = params['rate_drop']
                    )
hypermodel.fit(X_train,y_train,

               eval_set = [(X_train,y_train),(X_test,y_test)]
)
fin = time.perf_counter()

hypermodel.save_model('Project3/Regressors/tuned_xgboost.json')

print('REGRESSION REPORT:')
print(f'Hyperparameter search and fit performed in {fin-start} s')
