# FYSSTK4155

This is a project repository for codes developed in the course FYS-STK4155 at the University of Oslo. All code in project 1 and 2 is written in collaboration with
Ellen Reeka and Ines Santandreu. 

## Project 1
Implementation of Ordinary Least Squares (OLS), Ridge and LASSO regression for the purpose of studying topological data. 
## Project 2
Implementation of a feed forward neural network (FFNN) from scratch with backpropagation. Under Examples the code is used for both regression and binary classification. 
## Project 3
Implementation of Keras DNN and xgboost to predict thermal conductivity in inorganic materials. Implementation of regressor tuners are central to this project. 
#### load_datasets.py
Library: contains call to return scaled, normalized and train-test splitted data sets.
#### build_neuralnetwork.py
In this code, Keras is used to create a neural network builder and wrap the builder in a Keras Tuner engine for hyperparameter optimization. 

#### build_xgb_tuner.py
In this code, xgboost is used to call XGBRegressor(). The regressor is wrapped in a hyperparameter search function RandomizedSearchCV() from sklearn to determine the optimal
configuration of the XGBRegressor. After determining the optimal configuration, a final 'hypermodel' is called using this parameter setup and saved as the tuned regressor.

#### regression_analysis.py
Here each tuned regressor is loaded and studied with plots and metric computations.


  
