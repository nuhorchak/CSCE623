# -*- coding: utf-8 -*-
"""
Created on Thu May  4 22:48:59 2017

@author: nicho
"""
import numpy as np

#Validation set MSE graph ################################### 
val_MSE = np.array(val_MSE)
MSEsplitVAL = np.split(val_MSE,10)
MSEsplitVAL = np.array(MSEsplitVAL).T
MSEsplitVAL = pd.DataFrame(MSEsplitVAL)
for graph in range(0,10):
    plt.plot([1,2,3,4], MSEsplitVAL[graph])
    VALMean = MSEsplitVAL.mean(axis=1)
    VALmin = min(VALMean)
    minVALind = (VALMean==VALmin).argmax()+1
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
    K5mean = MSEsplitK5.mean(axis=1)
    K5min = min(K5mean)
    minK5ind = (K5mean==K5min).argmax()+1
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
    K10mean = MSEsplitK10.mean(axis=1)
    K10min = min(K10mean)
    minK10ind = (K10mean==K10min).argmax()+1
    plt.scatter(minK10ind, K10min,s=300, facecolors='none', edgecolors='r', label="Min MSE")
plt.xticks([1,2,3,4])
plt.xlabel('Model Order')
plt.ylabel('MSE')
plt.title("K Fold, K = 10")
plt.show()


