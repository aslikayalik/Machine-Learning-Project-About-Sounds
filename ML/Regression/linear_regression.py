# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 00:20:43 2023

@author: kayal
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from coreR import get_audio_duration, format_size
from coreR import X_train, y_train, X_val, y_val, X_test, y_test,y
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np


lr = LinearRegression()

lr.fit(X_train, y_train)

# Tahmin yapma test seti üzerinde
y_pred = lr.predict(X_test)
# R-kare değerini hesaplama
r2 = r2_score(y_test, y_pred)
print("Test veri seti için R-kare değeri:", r2)


# Validation veri seti üzerinde tahminler yapma
y_val_pred = lr.predict(X_val)
# R2 skorunu hesaplama
r2_val = r2_score(y_val, y_val_pred)
print("Validation veri seti için R-kare değeri:", r2_val)


print("Mean absolute error:", mean_absolute_error(y_test, y_pred))

print("Mean squared error:",np.sqrt(mean_squared_error(y_test, y_pred)))


# Tahmin yapma
ses_boyutu = [[5904052]]
tahmin = lr.predict(ses_boyutu)
print("Tahmin edilen süre:", tahmin)







    
    
