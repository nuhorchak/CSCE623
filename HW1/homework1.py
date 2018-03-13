# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 13:02:34 2017

@author: nuhorchak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


#load auto data into python
auto_data = pd.read_csv('auto.csv', na_values = '?')
#find missing data (missing = ?)
auto_data = auto_data.dropna(how='any')

#simple plot of horsepower v mpg
auto_data.plot(kind='scatter',x='horsepower',y='mpg')
plt.title('Horsepower vs MPG')
plt.show()
#simple plot of displacement v mpg
auto_data.plot(kind='scatter',x='displacement',y='mpg')
plt.title('Displacement vs MPG')
plt.show()
#simple plot of weight v mpg
auto_data.plot(kind='scatter',x='weight',y='mpg')
plt.title('Weight vs MPG')
plt.show()

#attempt at 3D scatter
X = auto_data['horsepower']
X = np.array(X)
Y = auto_data['displacement']
Y = np.array(Y)
Z = auto_data['mpg']
Z = np.array(Z)
fig = plt.figure(figsize=(7,5))
ax = Axes3D(fig)
line1 = ax.plot(X,Y,Z,'r+')
ax.set_xlabel('Horsepower')
ax.set_ylabel('Displacement')
ax.set_zlabel('MPG')
plt.show()


#assume that B0 is 40, estimate is fixed at 40, estimate B1 (eyeball method)
#from observation, the slope looks to be between 0 and -.2, so we will use that for our next part
B1_range = np.linspace(-.1,-.2)
print(B1_range)
RSS = np.zeros([50,1])


#create a vector of ones for multiplication with intercept, same length as the data set
X1 = np.ones(len(auto_data['horsepower']))
#given intercept
B0 = 40
#observed Y values
Y_VALS = auto_data['mpg']


for i in range(0,len(B1_range)):
    #select slope from slope range above
    B1 = B1_range[i]

    #create beta vector
    Beta = np.append(B0,B1)

    #create X vector from data
    X2 = auto_data['horsepower']
    X2 = np.array(X2)
    #combine vectors to make X_VAL for matrix multiplication
    X_VAL = np.c_[X1,X2]
    #print(X_VAL)
    #calculate Y_HAT, predicted values - note, make sure you use @ symbol for matrix multiplication
    Y_HAT = X_VAL@Beta
    #calculate error
    error = Y_VALS - Y_HAT
    error = np.array(error)
    #residual sum of squares for Y_VALS - Y_HAT
    NEW_RSS = error.T @ error
    #add each iterations calculated RSS to the array of RSS'
    RSS[i] = NEW_RSS
RSS_tot = np.array(RSS)
    
    
#graph slope as a fucntion of RSS
plt.plot(B1_range, RSS,'r.')
plt.xlabel(r'$\beta_1$')
plt.ylabel('RSS')
plt.show()

#new graph with updated slope estimate from last graph
B1_range = np.linspace(-.15838,-.15839)
plt.plot(B1_range, RSS,'r.')
plt.xlabel(r'$\beta_1$')
plt.ylabel('RSS')
plt.show()

#compute the slope of the line, using the x and plot it against the scatter plot
#B1 = -.15837
line = 40 + (auto_data['horsepower'] * -.158385)
auto_data.plot(kind='scatter',x='horsepower',y='mpg')
plt.title('Horsepower vs MPG')
plt.plot(auto_data['horsepower'], line, 'r-')
plt.show()

#create a closed form function to compute the beta coeffecients 
def computeBetas(xVec,yVec):
    
    x_bar = np.mean(xVec)
    y_bar = np.mean(yVec)
    diff_x = np.subtract(xVec, x_bar)
    diff_y = np.subtract(yVec,y_bar)
    sum_XY = diff_x @ diff_y.T
    sum_X_sq = np.sum(diff_x **2)
    beta_1 = sum_XY / sum_X_sq
    beta_0 = y_bar - (beta_1 * x_bar)
    betas = (beta_0,beta_1)
    #this call returns the TUPLE containing the values for coeffecients
    return betas


#using the auto data, compute B0 and B1
x_vec = auto_data['horsepower']
x_vec = np.array(x_vec)
y_vec = auto_data['mpg'];
y_vec = np.array(y_vec)
Beta_hat = computeBetas(x_vec,y_vec)
print(Beta_hat)
    

#make a new plot of the updated beta values and the scatter plot
#B1 = -.15837
line = Beta_hat[0] + (auto_data['horsepower'] * Beta_hat[1])
auto_data.plot(kind='scatter',x='horsepower',y='mpg')
plt.title('Horsepower vs MPG updated regression')
plt.plot(auto_data['horsepower'], line, 'r-')
plt.show()

#use sklearn's linear-model function to fit a linear model
regr = linear_model.LinearRegression()
x_vec = x_vec[:, np.newaxis]
y_vec = y_vec[:, np.newaxis]
regr.fit(x_vec,y_vec)

# The coefficients
print('Intercept: \n', regr.intercept_)
print('Slope: \n', regr.coef_)

# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(x_vec) - y_vec) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x_vec, y_vec))

#make a ne plot of the original scatter plot overlayed with the regression line
#from sklearn's linear model
learn_int = np.array(regr.intercept_).ravel()
learn_slope = np.array(regr.coef_).ravel()
line = learn_int + (auto_data['horsepower'] * learn_slope)
auto_data.plot(kind='scatter',x='horsepower',y='mpg')
plt.title('Horsepower vs MPG SKLEARN style')
plt.plot(auto_data['horsepower'], line, 'r-')
plt.show()



    
    