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
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, RandomizedSearchCV

from load_datasets import ricci, linear


training_set, testing_set = ricci()

X_train, y_train = training_set ; X_test, y_test = testing_set

regressor = xgb.XGBRFRegressor(booster = 'dart',
                             objective = 'reg:squarederror',
                             eval_metric = 'mae')

hyperspace = {
    'num_parallel_tree':[1,5,10],
    'min_child_weight':[50,100,500],
    'gamma':[i/10.0 for i in range(1,3)],
    'reg_alpha' : [1,10,40],
    'subsample':[i/10.0 for i in range(6,11)],
    'colsample_bytree':[i/10.0 for i in range(6,11)],
    'max_depth': [2,3,4,6,7],
    'eta': np.logspace(-4,-1,10),
    'skip_drop' : [0.5,0.8,1.],
    'rate_drop' : [0.5,0.6,0.7]
}

CVFolds = KFold(n_splits = 2)

grid = RandomizedSearchCV(regressor,
                    hyperspace,
                    scoring = 'neg_root_mean_squared_error',
                    n_iter = 200,
                    n_jobs = -1,
                    cv = CVFolds,
                    verbose = 3).fit(X_train,y_train)

params = grid.best_params_
print(params)

hypermodel = xgb.XGBRFRegressor(booster = 'dart',
                             objective = 'reg:squarederror',
                             eval_metric = 'mae',
                             num_parallel_tree = params['num_parallel_tree'],
                             min_child_weight = params['min_child_weight'],
                             gamma = params['gamma'],
                             reg_lambda = params['reg_alpha'],
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

hypermodel.save_model('./tuned_xgboost.json')
