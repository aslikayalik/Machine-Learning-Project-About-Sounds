# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 00:36:33 2023

@author: kayal
"""

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from coreR import get_audio_duration, format_size
from coreR import X_train, y_train, X_val, y_val, X_test, y_test,y


base_elastic_model = ElasticNet()

param_grid = {'alpha':[0.1,1,5,10,50,100],
             'l1_ratio':[.1,.5,.7,.9,.95,.99,1]}

grid_model = GridSearchCV(estimator=base_elastic_model, 
                         param_grid = param_grid,
                         scoring='neg_mean_squared_error',
                         cv=5,
                         verbose=1)

grid_model.fit(X_train, y_train)

print(grid_model.best_estimator_)

print(grid_model.best_params_)

y_pred = grid_model.predict(X_test)

print("Mean absolute error:", mean_absolute_error(y_test, y_pred))

print("Mean squared error:",np.sqrt(mean_squared_error(y_test, y_pred)))

print(np.mean(y))

# Tahmin yapma
ses_boyutu = [[5904052]]
tahmin = grid_model.predict(ses_boyutu)
print("Tahmin edilen süre:", tahmin)






