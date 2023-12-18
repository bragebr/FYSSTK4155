import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from keras.models import load_model

from load_datasets import ricci, linear

training_set, test_set = ricci()
X_test, y_test = test_set

tuned_keras_regressor = load_model('tuned_regressor.keras')

tuned_keras_regressor.summary()

y_pred = tuned_keras_regressor.predict(X_test)

print(np.abs(y_test - y_pred).mean())

def similarity_score(y_test,y_pred):
    similarity_score = np.zeros(len(y_test))


    for j in range(len(y_test)):

        absdiff = np.abs((y_test[j] - y_pred[j]))
        similarity_score[j] = absdiff

    return similarity_score


sns.jointplot(data=pd.DataFrame({'True Value':np.log(y_test).flatten(),
                                 'Predicted Value':np.log(y_pred).flatten()}),
              x="True Value",
              y="Predicted Value", kind="reg")
plt.show()

plt.figure(figsize=(5,5))
graph = plt.scatter(y_test,
                    y_pred,
                    c=similarity_score(y_test,y_pred),
                    cmap=plt.cm.RdBu)
cb = plt.colorbar(graph)
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_pred), max(y_test))
p2 = min(min(y_pred), min(y_test))

plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


booster = xgb.XGBRegressor()

booster.load_model('./tuned_xgboost.json')

print(booster.best_iteration)

y_pred = booster.predict(X_test)

print('Boost MAE', np.abs(y_test - y_pred).mean())

plt.figure(figsize=(5,5))
graph = plt.scatter(y_test,
                    y_pred,
                    c=similarity_score(y_test,y_pred),
                    cmap=plt.cm.RdBu)
cb = plt.colorbar(graph)
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_pred), max(y_test))
p2 = min(min(y_pred), min(y_test))

#plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()
