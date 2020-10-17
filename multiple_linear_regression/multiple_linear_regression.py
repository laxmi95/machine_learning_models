#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laxmi Garde

"""

# importing the libraries
import numpy as np
import matplotlib.pyplot as py
import pandas as pd


# importing the dataset
dataset = pd.read_csv('Startups.csv')
# X dependent variable present in the first 3 coloumns of the dataset
# Y independent variable 'profit' in the last column
X = dataset.iloc[: , :-1].values
Y = dataset.iloc[: , -1].values


# encode categorical data
# 'State' is categorical field in the dataset : use OneHotEncoding 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])] , remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)


# Splitting the dataset into the Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# train the Multiple Linear Regression model on Training set (0.8)
# No need to take care of dummy var trap or Backward elimination as sklearn takes care of this operation
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# predict the test results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
# display Predicted vs Real profit column as vectors.  axis = 1 for vertical vector display
print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))

