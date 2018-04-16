'''多元线性回归'''

from numpy import genfromtxt
import numpy as np
from sklearn import datasets, linear_model

dataPath = r'Delivery.csv'
deliveryData = genfromtxt(dataPath, delimiter=',')

print('data: %s' % deliveryData)

X = deliveryData[:, : -1]
Y = deliveryData[:, -1]
print('X: %s' % X)
print('Y: %s' % Y)

regr = linear_model.LinearRegression()
regr.fit(X, Y)
print('coefficients: %s' % regr.coef_)
print('intercept: %s' % regr.intercept_)

xPred = [[102, 6]]
yPred = regr.predict(xPred)
print('predicted y: %s' % yPred)