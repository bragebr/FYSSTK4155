import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
import scipy.stats as stats
from keras.models import load_model
from sklearn.metrics import r2_score, mean_absolute_error

from load_datasets import ricci, linear, get_full_ricci, get_full_linear

sns.set_theme()

# = = = = = = = = = = = = = = = = = = = = = = = = = =
"""
Code regression_analysis loads the tuned regressors
saved in build_xgb_tuner and build_neuralnetwork
for postprocessing and results production.

Explanation : the author notes that in this code, a
function similarity_score() is defined and it is
indeed just 1-MAE, however similarity_score reads more
intuitively in an infographic like a plot.

Function : _count() computes and prints a count of
how many predicted data points that commit an absolute
error lower than a set threshold. It also prints how
many percent of the total data set these counts
account for.
"""
# = = = = = = = = = = = = = = = = = = = = = = = = = =

def similarity_score(y_test,y_pred):
    similarity_score = np.zeros(len(y_test))


    for j in range(len(y_test)):

        absdiff = np.abs((y_test[j] - y_pred[j]))
        similarity_score[j] = 1 - absdiff

    return similarity_score

def count(y_test,y_pred):

    diff = np.abs(y_test - y_pred)
    thresholds = [10e-7,10e-6,10e-5]

    counts = np.zeros(len(thresholds))
    percentages = np.zeros(len(thresholds))

    for i in range(len(thresholds)):
        count = (diff < thresholds[i]).sum()
        counts[i] = count
        percentages[i] = count / len(diff)*100

    return counts, percentages



def PLOT_ANALYSIS(y_test,y_pred):

    slope,int = np.polyfit(y_test.flatten(),y_pred.flatten(),1)
    x_plot = np.linspace(np.sort(y_test.flatten())[1],np.max(y_test),100)

    fit_line = np.polyval([slope,int], x_plot)

    MAE = mean_absolute_error(y_test,y_pred)
    R2  = r2_score(y_test,y_pred)

    return x_plot, fit_line, MAE, R2


# load data for preditions and visualization
data = get_full_ricci()
X_test,y_test = data
# load saved regressor models
tuned_keras_regressor = load_model('Project3/Regressors/tuned_regressor.keras')
tuned_boost_regressor = xgb.XGBRegressor()
tuned_boost_regressor.load_model('Project3/Regressors/tuned_xgboost.json')

#tuned_keras_regressor.summary()

y_keras = tuned_keras_regressor.predict(X_test)
y_boost = tuned_boost_regressor.predict(X_test)

print(count(y_test,y_boost))


xgb.plot_importance(tuned_boost_regressor)
plt.show()

x_k, fit_k, MAE_k, R2_k = PLOT_ANALYSIS(y_test,y_keras)
x_b, fit_b, MAE_b, R2_b = PLOT_ANALYSIS(y_test,y_boost)



graph = plt.scatter(y_test,
                    y_keras,
                    c=similarity_score(y_test,y_keras),
                    cmap=plt.cm.viridis)
plt.fill_between(x_k,
                 fit_k-np.sqrt(MAE_k),
                 fit_k+np.sqrt(MAE_k),
                 color = 'orange',
                 label = r'+- $\sqrt{MAE}$',
                 alpha=0.3)

plt.fill_between(x_k,
                 fit_k-np.sqrt(MAE_k),
                 fit_k-2*np.sqrt(MAE_k),
                 color = 'red',
                 alpha=0.33)
plt.fill_between(x_k,
                 fit_k+np.sqrt(MAE_k),
                 fit_k+2*np.sqrt(MAE_k),
                 color = 'red',
                 label = r'+- $2\sqrt{MAE}$',
                 alpha=0.33)


textstr = '\n'.join((
    'Keras Model',
    r'$R^2=%.3f$' % (R2_k, ),
    r'MAE=%.3f' % (MAE_k, )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(-10,1.5,textstr,verticalalignment='top', bbox=props)

cbar = plt.colorbar(graph)
cbar.ax.set_ylabel('Similarity Score')
plt.legend()
plt.ylabel(r'Predictions')
plt.xlabel(r'True Values')
plt.legend()
plt.savefig('Project3/Figures/analysis_linear_keras_2.png')
plt.show()


graph2 = plt.scatter(y_test,
                     y_boost,
                     c=similarity_score(y_test,y_boost),
                     cmap=plt.cm.viridis)
plt.fill_between(x_b,
                 fit_b-np.sqrt(MAE_b),
                 fit_b+np.sqrt(MAE_b),
                 color = 'orange',
                 label = r'+- $\sqrt{MAE}$',
                 alpha=0.3)
plt.fill_between(x_b,
                 fit_b-np.sqrt(MAE_b),
                 fit_b-2*np.sqrt(MAE_b),
                 color = 'red',
                 alpha=0.33)

plt.fill_between(x_b,
                 fit_b+np.sqrt(MAE_b),
                 fit_b+2*np.sqrt(MAE_b),
                 color = 'red',
                 label = r'+- $2\sqrt{MAE}$',
                 alpha=0.33)
textstr = '\n'.join((
    'XGBoost Model',
    r'$R^2=%.3f$' % (R2_b, ),
    r'MAE=%.3f' % (MAE_b, )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(-10,1.5,textstr,verticalalignment='top', bbox=props)

cbar = plt.colorbar(graph2)
cbar.ax.set_ylabel('Similarity Score')
plt.legend()
plt.ylabel(r'Predictions')
plt.xlabel(r'True Values')

plt.savefig('Project3/Figures/analysis_ricci_boost_2.png')
plt.show()

#==============================
# Uncomment the code below if
# using the linear data set by
# function call linear()
#==============================


"""
plt.title(f'Taylor Series Approximation of an Exponential Function''\n'
'--------------------------------------''\n'
'Model Comparison via Keras and XGBoost')
plt.plot(xlinear,y_test,'bo',alpha=0.08)
plt.plot(xlinear,y_keras,linestyle = 'dashed', color='red', label = 'Keras DNN')
plt.plot(xlinear,y_boost, color='orange', label = 'XGBoost Tree')
plt.xlabel('x') ; plt.ylabel('y')
plt.legend() ; plt.savefig('Project3/Figures/benchmark.png')
plt.show()
"""
