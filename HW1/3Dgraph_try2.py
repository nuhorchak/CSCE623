# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 19:52:20 2017

@author: nuhorchak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale

#load auto data into python
auto_data = pd.read_csv('auto.csv', na_values = '?')
#find missing data (missing = ?)
auto_data = auto_data.dropna(how='any')

# Regression coefficients (Ordinary Least Squares)
X = (auto_data.horsepower).values.reshape(-1,1)
y = np.array(auto_data.mpg)

regr = linear_model.LinearRegression()
regr.fit(X,y)
print(regr.intercept_)
print(regr.coef_)


# Create grid coordinates for plotting
B0 = np.linspace(regr.intercept_-5, regr.intercept_+5, 50)
slope = regr.coef_
B1 = np.linspace(slope-.2, slope+.2,50)
xx, yy = np.meshgrid(B0, B1, indexing='xy')
Z = np.zeros((B0.size,B1.size))

# Calculate Z-values (RSS) based on grid of coefficients
for (i,j),v in np.ndenumerate(Z):
    Z[i,j] =((y - (xx[i,j]-X.ravel()*yy[i,j]))**2).sum()/1000

# Minimized RSS
min_RSS = r'$\beta_0$, $\beta_1$ for minimized RSS'
min_rss = np.sum((regr.intercept_+slope*X - y.reshape(-1,1))**2)/1000
min_rss

fig = plt.figure(figsize=(15,6))
fig.suptitle('RSS - Regression coefficients', fontsize=20)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# Left plot
CS = ax1.contour(xx, yy, Z, cmap=plt.cm.Set1)#, levels=[2.15, 2.2, 2.3, 2.5, 3])
ax1.scatter(regr.intercept_, regr.coef_[0], c='r', label=min_RSS)
ax1.clabel(CS, inline=True, fontsize=10, fmt='%1.1f')

# Right plot
ax2.plot_surface(xx, yy, Z, rstride=3, cstride=3, alpha=0.3)
ax2.contour(xx, yy, Z, zdir='z', offset=Z.min(), cmap=plt.cm.Set1,
            alpha=0.4)#, levels=[2.15, 2.2, 2.3, 2.5, 3])
ax2.scatter3D(regr.intercept_, regr.coef_[0], min_rss, c='r', label=min_RSS)
ax2.set_zlabel('RSS')
ax2.set_zlim(Z.min(),Z.max())
ax2.set_ylim(-0.13,-0.17)

# settings common to both plots
for ax in fig.axes:
    ax.set_xlabel(r'$\beta_0$', fontsize=17)
    ax.set_ylabel(r'$\beta_1$', fontsize=17)
    #ax.set_yticks([0.03,0.04,0.05,0.06])
    ax.legend()