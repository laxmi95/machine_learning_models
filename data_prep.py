#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: laxmi garde
"""


#starting with data preprocessing

#1. importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#2. importing datasets
# x : features used for predicition (independent variables)
# y : dependent variables
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)


#3. handling empty data fields : using SimpleImputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)


#4. Encode data
#4.1 encode independent variables : OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])] , remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)

#4.2 encode dependent variables : LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)


#5. feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
print(x)


#6. Split training & test sets for later
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state = 0)
print('Training set:')
print(x_train)
print(y_train)
print('Test set:')
print(x_test)
print(y_test)






