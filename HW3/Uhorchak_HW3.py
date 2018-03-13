# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 21:23:36 2017

@author: nuhorchak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


#provided code
#1.	You will generate datasets using the following function.  Include this code early in your python 
#file so you can call it in later code chunks.  
#Note that in this data, x is a predictor / feature and y is a response variable. 
def makeData(myseed):
    np.random.seed(myseed)   #dont forget to import numpy as np
    #y = np.zeros(shape = 100) #100  0-1 normal floats
    x = np.random.normal(size=100) #100  0-1 normal floats
    y = x-2*(x**2)+np.random.normal(size = 100)
    df = pd.DataFrame({'x':x, 'y':y})
    return(df) 

#2.	Call the function to make a dataset:  df1=makeData(1).  Answer the following questions:  
#How many observations?  How many features/predictors?  
#Determine and display the value of n (count the observations) 
#and the value for p (count the predictors) in this dataset  
df1=makeData(1)
[n,p] = df1.shape
print("observations")
print(n)
print("Predictors")
print(p)

#3. 3.	Create a scatterplot of X against Y.  Describe the shape of the data.  
#What kind of relationship will fit the data?  Linear?  Polynomial (and if so, what order of polynomial)?  
#Form an official hypothesis and state it.
##################################graph##########################################################
plt.figure()
plt.scatter(df1.x, df1.y)
plt.title('Data Set 1')
plt.xlabel('Predictor')
plt.ylabel('Response')
#4. 4.	Implement this helper function early in your code – 
#it generates a polynomial design matrix from a single feature vector x.  
#The return matrix contains columns of x**0, x**1, … x**p where p is the desired highest order of the polynomial.  
#Note that since it returns a design matrix, the columns correspond to B0 through Bp

def polyDesignMatrix(x, p):
    x = np.array(x)  # dont forget to import numpy as np
    X = np.transpose(np.vstack((x**k for k in range(p+1))))
    return(X)

#5. 5.	Create a function to run LOOCV to return cross-validation performance on a 
#linear regression model with polynomial terms

def LOOCVerr(df, modelOrder):

    polyX = polyDesignMatrix(df.x, modelOrder)
    regr = LinearRegression()
    regr.fit(polyX, df.y)
    y_hat = regr.predict(polyX)
    
    #using the LOOCV shortcut from page 180
    xbar = np.mean(polyX, axis=0)
    xsum = np.sum(np.power(polyX - xbar, 2))
    nrows = np.shape(polyX)[0]
    mses = []
    for row in range(0, nrows):
        hi = (1 / nrows) + (np.sum(polyX[row] - xbar) ** 2 / xsum)
        mse = (df.y[row] - y_hat[row]) ** 2 / (1 - hi)
        mses.append(mse)
    return mses

#6.	Using df1 (where you ran makeData with a seed of 1) build a for-loop to run LOOCV to 
#generate error vectors using modelOrder values from 1 through 4.  
#LOOCV will build and return squared error vectors for 4 separate models which were
# evaluated with linear, linear+quadratic, linear+quadratic+cubic, 
#and linear+quadratic+cubic+quartic terms. 

#index = [np.arange(0,100,1)]
#columns = ['1','2','3','4']
err_vecs = pd.DataFrame()

for i in range(1,5):
    err = LOOCVerr(df1, i)
    err_vecs[i-1] = err
    
#7.	Plot the results from your LOOCV on models of order 1 through 4.  
#This plot should have the model order on the x axis and mean squared error on the y axis
# (MSE is the mean of the squared values of the error terms on the y axis).  
#Determine the model order with the minimum cross-validation MSE and indicate the minimizing 
#model order on the plot & report it. 
##################################graph##########################################################
x_vecs = [1,2,3,4]
y_vecs = [err_vecs[0].mean(), err_vecs[1].mean(),err_vecs[2].mean(),err_vecs[3].mean()]
plt.figure()
plt.plot(x_vecs, y_vecs)
plt.title("Polynomial Order v MSE")
plt.xticks(x_vecs)
plt.xlabel('Polynomial Order')
plt.ylabel('MSE')
#plot model order with min MSE and graph
min_MSE = min(y_vecs)
min_ind = y_vecs.index(min_MSE) +1
plt.scatter(min_ind, min_MSE, facecolors='none', edgecolors='r', label="Min MSE")
plt.legend()
plt.show()

#8. Now you will explore the results of model fittings by simulating different 
#samples of the original population.  Using a nested for loop, generate datasets using 
#10 different seeds (1 through 10).   Then obtain LOOCV squared errors on model orders 1 through 4.  
#After you receive each error term vector, compute the MSE as a scalar value.  
#Store the results in a new dataframe which has 3 columns (“myseed”, “modelOrder”, and “MSE”).  

MSEdf = pd.DataFrame()
myseed = []
modelOrder = []
MSE = []
for seeds in range(1, 11):
    data_sets = makeData(seeds)
    for model in range(1,5):
        MSE.append(np.mean(LOOCVerr(data_sets,model)))
        modelOrder.append(model)
        myseed.append(seeds)
for cols,data in zip(["myseed","modelOrder", "MSE"],[myseed, modelOrder, MSE]):
    MSEdf[cols] = data
    
#9.	Make a new “spaghetti plot” (like book figure 5.4 right side subplot) using the new dataframe 
#you created in step 7.  The x axis represents the model order and the y axis represents
# the avgError.  Each of the 10 lines in your plot should represent the LOOCV MSE over model 
# order for one dataset generated from a specific seed.

MSE = np.array(MSE)
MSEsplit = np.split(MSE,10)
MSEsplit = np.array(MSEsplit).T
MSEsplit = pd.DataFrame(MSEsplit)
for graph in range(0,10):
    plt.plot([1,2,3,4], MSEsplit[graph])
plt.xticks([1,2,3,4])
plt.xlabel('Model Order')
plt.ylabel('MSE')
plt.show()

#10.	Design, code, and execute a technique for (mathematically) determining which order of 
#model has the smallest total LOOCV error.  Report which order of model had the smallest total LOOCV error. 
# Is that what you expected in your hypothesis?  

MSEmean = MSEsplit.mean(axis=1)
minMSEind = MSEmean.idxmin(MSEmean)+1
print("LOOCV")
print('Min MSE error',minMSEind)

#11.	Build another function to perform validation using the “validation set approach” described in
# section 5.1.1 where a randomly-selected half of the dataset is used for training, and the 
#remaining portion is used for validation.  Your function should have the signature 
#VALSETerr(df,modelOrder,splitseed) and it should return a SINGLE MSE value of the prediction 
#quality on the validation set.  The randomness should be repeatable, based on controlling the random 
#seed in the data permutation before the split using splitseed.

def VALSETerr(df,modelOrder,splitseed):
    train = df.sample(frac=0.5, random_state=splitseed)
    test = df.drop(train.index)
    train = train.reset_index()
    test = test.reset_index()
    y_train = train['y']
    y_test = test['y']
    x_train = train['x']
    x_test = test['x']
    x_train = polyDesignMatrix(x_train,modelOrder)
    x_test = polyDesignMatrix(x_test,modelOrder)
    VALregr = LinearRegression()
    VALregr.fit(x_train, y_train)
    y_hat = VALregr.predict(x_test)
    MSE = mean_squared_error(y_test,y_hat)
    
    return MSE

#12.	Build another function to perform k-fold cross validation as described in the book 
#section 5.1.3.  This function will have the signature KFOLDerr(df,modelOrder,k,splitseed) 
#and will return a k-length vector of total-error terms. Each total-error term represents the 
#mean of the MSEs computed on each of the k folds. Membership of the data in each fold should be 
#determined randomly.  Hint:  When partitioning the data into folds be careful to write code that 
#handles non-integer fold-sizes appropriately.  The randomness should be repeatable, based on 
#controlling the random seed in the data permutation before the determination of the fold
# memberships using splitseed.

def KFOLDerr(df,modelOrder,k,splitseed):
    if k <= 1:
        return print("k folds should be >1")
    else:
        np.random.seed(splitseed)
        df = df.sample(frac=1)
        splitDf = np.array_split(df,k)
        #reset = deepcopy(polyDF)
        
        MSE = []
        for frames in range(k):
            testDF = splitDf[frames]
            trainDF = df.drop(testDF.index)
            x_train, x_test = trainDF.x.values, testDF.x.values
            y_train, y_test = trainDF.y.values, testDF.y.values
            x_train = polyDesignMatrix(x_train,modelOrder)
            x_test = polyDesignMatrix(x_test,modelOrder)
            Kregr = LinearRegression()
            Kregr.fit(x_train, y_train)
            y_hat = Kregr.predict(x_test)
            MSE.append(mean_squared_error(y_hat, y_test))
    return MSE


#13.	In a later step we will visualize the reliability of 3 validation methods:
#A) validation set; B) 5-fold cross-validation; C) 10-fold cross-validation. Write code to
#compute and store the MSEs of each of the 3 validation methods (A, B, C) for each model order (1 through 4) on 
#splitseed values of 1 through 10.  You are collecting a total of 
#3 x 4 x 10 = 120 MSE values in this step.


val_MSE = []
k5_MSE = []
k10_MSE = []
for seed in range(1,11):
    df = makeData(seed)
    for order in range(1,5):
        val_MSE.append(np.mean(VALSETerr(df,order, seed)))
        k5_MSE.append(np.mean(KFOLDerr(df,order,5,seed)))
        k10_MSE.append(np.mean(KFOLDerr(df,order,10,seed)))

        
#14.	Make 3 “spaghetti plots” – one for each validation method (validation set, 5-fold, and 10-fold).  
#In these plots, the X axis is model order and the Y axis is MSE.  In each spaghetti plot there will be 10 lines 
#(1 line per random seed which controlled the data split into train/val partitions).  
#Each of the 10 seed lines will have 4 model order points which display the MSE at each of those model orders. 
# For each line in a spaghetti plot, annotate the point with the lowest MSE using a datamarker 
#(there will be one point indicated on each line).    

#Validation set MSE graph ################################### 
val_MSE = np.array(val_MSE)
MSEsplitVAL = np.split(val_MSE,10)
MSEsplitVAL = np.array(MSEsplitVAL).T
MSEsplitVAL = pd.DataFrame(MSEsplitVAL)
for graph in range(0,10):
    plt.plot([1,2,3,4], MSEsplitVAL[graph])
    VALmin = min(MSEsplitVAL[graph])
    minVALind = (MSEsplitVAL[graph]==VALmin).argmax()+1
    plt.scatter(minVALind, VALmin,s=300, facecolors='none', edgecolors='r', label="Min MSE")
plt.xticks([1,2,3,4])
plt.xlabel('Model Order')
plt.ylabel('MSE')
plt.title("Validation Set")
plt.show()

#K Fold, K = 5 graph #######################################
plt.figure()
k5_MSE = np.array(k5_MSE)
MSEsplitK5 = np.split(k5_MSE,10)
MSEsplitK5 = np.array(MSEsplitK5).T
MSEsplitK5 = pd.DataFrame(MSEsplitK5)
for graph in range(0,10):
    plt.plot([1,2,3,4], MSEsplitK5[graph])
    K5min = min(MSEsplitK5[graph])
    minK5ind = (MSEsplitK5[graph]==K5min).argmax()+1
    plt.scatter(minK5ind, K5min,s=300, facecolors='none', edgecolors='r', label="Min MSE")
plt.xticks([1,2,3,4])
plt.xlabel('Model Order')
plt.ylabel('MSE')
plt.title("K Fold, K = 5")
plt.show()

#K Fold, K = 10 graph ########################################
plt.figure()
k10_MSE = np.array(k10_MSE)
MSEsplitK10 = np.split(k10_MSE,10)
MSEsplitK10 = np.array(MSEsplitK10).T
MSEsplitK10 = pd.DataFrame(MSEsplitK10)
for graph in range(0,10):
    plt.plot([1,2,3,4], MSEsplitK10[graph])
    K10min = min(MSEsplitK10[graph])
    minK10ind = (MSEsplitK10[graph]==K10min).argmax()+1
    plt.scatter(minK10ind, K10min,s=300, facecolors='none', edgecolors='r', label="Min MSE")
plt.xticks([1,2,3,4])
plt.xlabel('Model Order')
plt.ylabel('MSE')
plt.title("K Fold, K = 10")
plt.show()

#15.	Using your plots from the previous step, determine which of the validation techniques 
#(validation set, 5-fold, and 10-fold) is most reliable for choosing model order on this dataset, 
#and discuss your answer & reasoning. 

#16.	Implement code for determining the overall best-order model from each of the 4 validation methods 
#(LOOCV, validation set, 5-fold and 10-fold crossval).  Report the best-order model chosen by each validation method.  
#Also, select what you believe to be the best overall model order based on the information provided by the 4 validation methods.   
        
#val set min MSE
print("Validation Set")
print('Min MSE model: ',minVALind)
print('Min MSE error: ',VALmin)
#k=5 min MSE
print("K = 5 K Fold")
print('Min MSE model: ',minK5ind)
print('Min MSE error: ',K5min)
#k=10 min MSE
print("K = 10 K Fold")
print('Min MSE model: ',minK10ind)
print('Min MSE error: ',K10min)