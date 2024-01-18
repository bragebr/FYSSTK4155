import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor

from load_datasets import ricci, linear

training_set, testing_set = ricci()

X_train, y_train = training_set ; X_test, y_test = testing_set

regressor = GradientBoostingRegressor(loss = 'absolute_error')

hyperspace = {
    'n_estimators':[1,5,10],

    'subsample':[i/10.0 for i in range(6,11)],
    'max_depth': [2,3,4,6,7],
    'learning_rate': np.logspace(-4,2,10),

}

CVFolds = KFold(n_splits = 10)

grid = RandomizedSearchCV(regressor,
                    hyperspace,
                    scoring = 'neg_root_mean_squared_error',
                    n_iter = 20,
                    n_jobs = -1,
                    cv = CVFolds,
                    verbose = 3).fit(X_train,y_train)

params = grid.best_params_
print(params)

hypermodel = GradientBoostingRegressor(booster = 'dart',
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

hypermodel.save_model('Project3/Regressors/tuned_gb.json')
