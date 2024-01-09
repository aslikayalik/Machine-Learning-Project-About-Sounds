# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 00:09:16 2023

@author: kayal
"""

from sklearn.tree import DecisionTreeClassifier
from core import features_extractor 
from core import X_train, y_train, X_val, y_val, X_test, y_test
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_val_pred = dt.predict(X_val)
y_pred = dt.predict(X_test)

val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_pred)

print("Validation accuracy:", val_accuracy)
print("\n Test accuracy: ", test_accuracy)
print("\n Decision Tree Confusion matrix :\n\n", confusion_matrix(y_test, y_pred))
print("\n Decision Tree Classification report :\n\n" ,classification_report(y_test, y_pred))

# Tahmin :
filename = 'C:\\Users\\kayal\\ML\\07070171.wav'
features = features_extractor(filename)
features = features.reshape(1,-1)
predict_data = dt.predict(features)

print("Tahmin sonucu : ", predict_data)