import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, RandomizedSearchCV

from load_datasets import ricci, linear


training_set, testing_set = ricci()

X_train, y_train = training_set ; X_test, y_test = testing_set

regressor = xgb.XGBRegressor(booster = 'dart',
                             objective = 'reg:squarederror',
                             eval_metric = 'mae')

hyperspace = {
    'n_estimators':[10,40,50],
    'min_child_weight':[4,5],
    'gamma':[i/10.0 for i in range(1,10)],
    'subsample':[i/10.0 for i in range(6,11)],
    'colsample_bytree':[i/10.0 for i in range(6,11)],
    'max_depth': [2,3,4,6,7],
    'eta': [i/10.0 for i in range(1,6)],
    'skip_drop' : [0.5,0.8,1.],
    'rate_drop' : [0.5,0.6,0.7]
}

CVFolds = KFold(n_splits = 3, shuffle=True, random_state=42)

grid = RandomizedSearchCV(regressor,
                    hyperspace,
                    scoring = 'neg_root_mean_squared_error',
                    n_iter = 100,
                    n_jobs = -1,
                    cv = CVFolds,
                    verbose = 3).fit(X_train,y_train)

params = grid.best_params_

hypermodel = xgb.XGBRegressor(booster = 'dart',
                             objective = 'reg:squarederror',
                             eval_metric = 'mae',
                             n_estimators = params['n_estimators'],
                             min_child_weight = params['min_child_weight'],
                             gamma = params['gamma'],
                             subsample = params['subsample'],
                             colsample_bytree = params['colsample_bytree'],
                             max_depth = params['max_depth'],
                             eta = params['eta'],
                             skip_drop = params['skip_drop'],
                             rate_drop = params['rate_drop'],
                             eval_set = [(X_train,y_train),(X_test,y_test)],
                             early_stopping_rounds = 0.25*params['n_estimators'])

hypermodel.fit(X_train,y_train)

hypermodel.save_model('./tuned_xgboost.json')
