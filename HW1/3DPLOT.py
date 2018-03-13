# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:49:08 2017

@author: nuhorchak
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.preprocessing import scale


#load auto data into python
auto_data = pd.read_csv('auto.csv', na_values = '?')
#find missing data (missing = ?)
auto_data = auto_data.dropna(how='any')

#create X vector from data
X1 = np.ones(len(auto_data['horsepower']))
X2 = auto_data['horsepower']
X2 = np.array(X2)
X_VAL = np.c_[X1,X2]
Y_VALS = auto_data['mpg']
Y_VALS = np.array(Y_VALS)


#OPTIONAL RSM PLOT section
beta0_RSM = np.linspace(40,40.1,num=100)
beta1_RSM = np.linspace(-.105,-.205, num=100)

betas = np.array([39.5, -.14])
RSS_tot = np.zeros(shape=(100,100))

for i in range(100):
    betas[0] = beta0_RSM[i]
    for j in range(100):
        betas[1] = beta1_RSM[j]      
        #Y_HAT = X_VAL@betas
        #error = Y_VALS - Y_HAT
        #NEW_RSS = error.T @ error
        RSS_tot[i,j] = errors


        
fig = plt.figure()
ax = fig.gca(projection='3d')
beta0, beta1 = np.meshgrid(beta0_RSM,beta1_RSM, indexing='xy')
surf = ax.plot_surface(beta0,beta1,RSS_tot, rstride=1, cstride=1, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)

#ax.zaxis.set_major_locator(LinearLocator(10))
fig.colorbar(surf, shrink=0.5,aspect=5)

ax.set_xlabel('Beta0')
ax.set_ylabel('Beta1')
ax.set_zlabel('RSS')
#ax.set_zlim(RSS_tot.min(),RSS_tot.max())
#ax.set_ylim(-0.13,-0.17)

plt.show()
