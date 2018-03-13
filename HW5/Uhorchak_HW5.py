# -*- coding: utf-8 -*-
"""
Created on Fri May 19 13:24:36 2017

@author: NUhorcha
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.metrics import mean_squared_error


#1. Load data, remove those with no salary data, log transform the salaries
state = 1
Hitters = pd.read_csv('ISLR_Hitters.csv',index_col=0)
Hitters = Hitters.dropna(how='any')
Hitters = Hitters.reset_index()
Hitters['Salary'] = np.log(Hitters.Salary)
Hitters = Hitters.drop(['index'],axis=1)
Hitters = Hitters.drop(['League'],axis=1)
Hitters = Hitters.drop(['Division'],axis=1)
Hitters = Hitters.drop(['NewLeague'],axis=1)

#2. explore the data
corr_hitter = Hitters.corr()
#sns.heatmap(corr_hitter)
sns.heatmap(corr_hitter, annot=True, linewidths=.5)
plt.figure()
plt.scatter(Hitters.CHits, Hitters.Salary)
plt.xlabel('Hits')
plt.ylabel('Salary')
plt.show()
plt.figure()
plt.scatter(Hitters.CRuns, Hitters.Salary)
plt.xlabel('Runs')
plt.ylabel('Salary')
plt.show()
plt.figure()
plt.scatter(Hitters.CAtBat, Hitters.Salary)
plt.xlabel('At Bats')
plt.ylabel('Salary')
plt.show()
plt.figure()
plt.scatter(Hitters.CRBI, Hitters.Salary)
plt.xlabel('RBIs')
plt.ylabel('Salary')
plt.show()

#3. explination
# At bat, hits, home runs, runs, RBIs walks have the highest correlation with salary
#one would expect that pruning the tree depth to include only those facors would produce 
#reasonable results

#4. create training set of first 200 observations, test set of the remaining
Train = Hitters.loc[range(0,200),:]
Test = Hitters.loc[range(200,263),:]

y_train = Train['Salary']
x_train = Train
x_train = x_train.drop(['Salary'],axis=1)

y_test = Test['Salary']
x_test = Test
x_test = x_test.drop(['Salary'],axis=1)



#5. perform boosting
learn_rate = np.linspace(0.1,1,num=10)
max_depth = np.linspace(1,10,num=10)
OOB_improve = np.zeros((len(max_depth), len(learn_rate), 1000))
scores = np.zeros((len(max_depth), len(learn_rate), 1000))
i=0
j=0

#6 ? maybe?
for d in (max_depth):
    
    for l in (learn_rate):

        gbr = GBR(learning_rate = l, n_estimators = 1000, random_state = state, subsample = .75, max_depth = d)
        gbr.fit(x_train, y_train)
        #collect metrics
        scores[i,j,:] = gbr.train_score_
        OOB_improve[i,j,:] = 10**gbr.oob_improvement_
        j+=1
    i+=1
    j=0
    
#7 ish...
x_range = np.arange(1,1001)
for d in (max_depth):
    for k in range(1,len(learn_rate)):
        plt.plot(x_range, scores[d-1,k,:])
plt.show()

plt.figure()
for d in (max_depth):
    for k in range(1,len(learn_rate)):
        plt.plot(x_range, OOB_improve[d-1,k,:])
plt.show()
        
#8 find best depth of boosted trees


