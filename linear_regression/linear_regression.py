#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laxmi Garde

"""

#1. data preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 0)
print(x_train)
print(y_train)
print(x_test)
print(y_test)


#2. train simple linear regression model on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


#3. predict test set result
y_pred = regressor.predict(x_test)


#4. visualising the training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#5. visualising the test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()