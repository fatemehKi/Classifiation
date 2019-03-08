# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 18:32:09 2019

@author: Kian
"""
# Logistic Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#Fitting multiple linear regression on the training set
from sklearn.linear_model import LogisticRegression
Classifier = LogisticRegression(random_state=0)
Classifier.fit(X_train, y_train)

#predicting the teat set result
y_pred = Classifier.predict(X_test)

#Making the confusion matrix (to check the performance of the model)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
