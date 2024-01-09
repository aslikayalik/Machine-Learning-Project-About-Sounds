# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 00:04:32 2023

@author: kayal
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from core import features_extractor 
from core import X_train, y_train, X_val, y_val, X_test, y_test
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report , accuracy_score


knn = KNeighborsClassifier()

scaler= StandardScaler()
operations=[('scaler',scaler),('knn',knn)]
pipe = Pipeline(operations)

k_values = list(range(1,30))
param_grid = {'knn__n_neighbors':k_values}
full_cv_classifier = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
full_cv_classifier.fit(X_train, y_train)
full_cv_classifier.best_estimator_.get_params()
full_cv_classifier.cv_results_['mean_test_score']
scores = full_cv_classifier.cv_results_['mean_test_score']
plt.plot(k_values, scores, 'o-')
plt.xlabel("K")
plt.ylabel("Accuracy")

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_val_pred = knn.predict(X_val)
y_pred = knn.predict(X_test)

val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_pred)

print("Validation accuracy:", val_accuracy)
print("\n Test accuracy: ", test_accuracy)
print("\n KNN Confusion matrix :\n\n", confusion_matrix(y_test, y_pred))
print("\n KNN Classification report :\n\n" ,classification_report(y_test, y_pred))

# Tahmin :
filename = 'C:\\Users\\kayal\\ML\\07070171.wav'
features = features_extractor(filename)
features = features.reshape(1,-1)
predict_data = knn.predict(features)

print("Tahmin sonucu : ", predict_data)