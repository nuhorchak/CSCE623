# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 23:47:32 2017

@author: nicho
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression as LOGR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from patsy import dmatrices

#1. load auto data into python
data1 = pd.read_csv('datset1.csv', usecols=['0','1', '2'])
data2 = pd.read_csv('datset2.csv', usecols=['0','1', '2'])
data3 = pd.read_csv('datset3.csv', usecols=['0','1', '2'])

#2.	Explore the dataset by plotting the data points from both classes 
#as a function of X1 (x-axis) and X2 (y-axis) scores in colors 
#according to their labels (for example, one class is red, the other class is blue)

class1 = data1['1']
plt.hist(class1)
plt.title('Set 1')
plt.show()

class2 = data2['1']
plt.hist(class2)
plt.title('Set 2')
plt.show()

class3 = data3['1']
plt.hist(class3)
plt.title('Set 3')
plt.show()


