#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: laxmi garde
polynomial linear regression model
"""

#1. importing libraries & dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[: , -1].values

#2. Training linear regression model on entire dataset (to compare with polynomial linear reg. model)
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)

#3. Visualize linear regression model results
plt.scatter(X , Y , color = 'red')
plt.plot(X , linear_regressor.predict(X) , color = 'blue')
plt.title('Linear Regression Results')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#4. Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 4)
X_poly = poly_regressor.fit_transform(X)
linear_regressor1 = LinearRegression()
linear_regressor1.fit(X_poly , Y)

#5. Visualize Polynomial regression model results
plt.scatter(X , Y , color = 'red')
plt.plot(X , linear_regressor1.predict(X_poly) , color = 'blue')
plt.title('Polynominal Regression Results')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#6. Predicting a new result with Linear Regression
linear_regressor.predict([[6.5]])

#7. Predicting a new result with Polynomial Regression
linear_regressor1.predict(poly_regressor.fit_transform([[6.5]]))