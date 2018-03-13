# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 08:54:15 2017

@author: nuhorchak
"""

import pandas as pd
import numpy as np
import matplotlib as plt


#X matrix of grades from midterm
X = np.matrix('1 95; 1 85; 1 80; 1 70; 1 60')
#print(X)
#B1 is the matrix of linear B's - B0 and B1
B1 = np.matrix('22.6;.70')
Y = np.matrix('85;95;70;65;70')
#print(B1)
#Y_HAT is the predicted final grade from the model
Y_HAT = X@B1
print("Y_HAT")
print(Y_HAT)
#error is the residuals from true value - Y_HAT
error = Y - Y_HAT
print("ERROR")
print(error)
#residual sum of squares for Y - Y_HAT
RSS = error.T * error
print("RSS")
print(RSS)
#mean square error
MSE = RSS / len(error)
print("MSE")
print(MSE)
print("SE")
print(np.sqrt(MSE))
#predict a new final grade from the linear model that we created above, for a midterm grade of 80
midterm = np.matrix('1 80')
pred_final = midterm * B1
print("predicted final grade")
print(pred_final)