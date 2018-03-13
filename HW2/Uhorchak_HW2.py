# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 21:25:07 2017

@author: nuhorchak
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

class1 = data1['2']
X11 = data1['0']
X21 = data1['1']
plt.scatter(X11, X21, c=class1)
plt.title('Data Set 1')
plt.show()


class2 = data2['2']
X12 = data2['0']
X22 = data2['1']
plt.scatter(X12, X22, c=class2)
plt.title('Data Set 2')
plt.show()
 
class3 = data3['2']
X13 = data3['0']
X23 = data3['1']
plt.scatter(X13, X23, c=class3)
plt.title('Data Set 3')
plt.show()

stats = data1.groupby('2').corr()
print('Correlation matrix')
print(stats)
stats2 = data2.groupby('2').corr()
print(stats2)
stats3 = data3.groupby('2').corr()
print(stats3)

stats = data1.groupby('2').cov()
print('Covariance matrix')
print(stats)
stats2 = data2.groupby('2').cov()
print(stats2)
stats3 = data3.groupby('2').cov()
print(stats3)

#4. make a function to split the data in half
#data is brought in as a data set
#split is a number of data points to split the training set into
def Test_Train(data, split):
    #ensure even distribution by splitting by 0 and 1
    upper_data = data[data['2'] == 0]
    lower_data = data[data['2'] == 1]
    #generate random split of 0 and 1 data, by specified fraction
    train_Upper = upper_data.sample(frac=(split/len(data)))
    test_Upper = upper_data.drop(train_Upper.index)
    
    train_Lower = lower_data.sample(frac=split/len(data))
    test_Lower = lower_data.drop(train_Lower.index)
    #final merge of uniform splt of data
    train_df = [train_Upper,train_Lower]
    test_df = [test_Upper,test_Lower]
    train = pd.concat(train_df)
    test = pd.concat(test_df)  
    return train, test

[train11, test11] = Test_Train(data1,300)
[train22, test22] = Test_Train(data2,300)
[train33, test33] = Test_Train(data3,300)

#5. fit a model for each classifier using only the training set
#change column names of the data set
#strip extra columns and turn dataframe into array for analysis
###############LOGISTIC REGRESSION##############################
#dataset 1
train11 = train11.rename(columns = {'0':'X1train',
                        '1':'X2train',
                        '2':'y1train'})
y1, X1 = dmatrices('y1train ~ X1train + X2train', train11, return_type="dataframe")
y1=np.ravel(y1)
X1 = X1.reset_index().values
X1 = np.delete(X1, [0,1], axis=1)
log1 = LOGR()
log1 = log1.fit(X1,y1)
#model score
print('LOGR1 accuracy')
print(log1.score(X1, y1))
#average y value
#print(y1.mean())

#dataset 2
train22 = train22.rename(columns = {'0':'X1train',
                        '1':'X2train',
                        '2':'y1train'})
y2, X2 = dmatrices('y1train ~ X1train + X2train', train22, return_type="dataframe")
y2=np.ravel(y2)
X2 = X2.reset_index().values
X2 = np.delete(X2, [0,1], axis=1)
log2 = LOGR()
log2 = log2.fit(X2,y2)
#model score
print('LOGR2 accuracy')
print(log2.score(X2, y2))
#average y value
#print(y2.mean())

#dataset 3
train33 = train33.rename(columns = {'0':'X1train',
                        '1':'X2train',
                        '2':'y1train'})
y3, X3 = dmatrices('y1train ~ X1train + X2train', train33, return_type="dataframe")
y3=np.ravel(y3)
X3 = X3.reset_index().values
X3 = np.delete(X3, [0,1], axis=1)
log3 = LOGR()
log3 = log3.fit(X3,y3)
#model score
print('LOGR3 accuracy')
print(log3.score(X3, y3))
#average y value
#print(y2.mean())

##################################LDA################################
#data set 1

lda1 = LDA(n_components=2)
lda1 = lda1.fit(X1, y1)
print('LDA1 accuracy')
print(lda1.score(X1,y1))

#data set 2
lda2 = LDA(n_components=2)
lda2 = lda2.fit(X2, y2)
print('LDA2 accuracy')
print(lda2.score(X2,y2))

#data set 3
lda3 = LDA(n_components=2)
lda3 = lda3.fit(X3, y3)
print('LDA3 accuracy')
print(lda3.score(X3,y3))
##################################QDA################################

#data set 1
qda1 = QDA(tol=0.1)
qda1 = qda1.fit(X1, y1)
print('QDA1 accuracy')
print(qda1.score(X1,y1))

#data set 2
qda2 = QDA(tol=0.1)
qda2 = qda2.fit(X2, y2)
print('QDA2 accuracy')
print(qda2.score(X2,y2))

#data set 3
qda3 = QDA(tol=0.1)
qda3 = qda3.fit(X3, y3)
print('QDA3 accuracy')
print(qda3.score(X3,y3))

#6. for each trained classifier, use the test set to determine the probabilities for which each
#classifier believes the dataset belongs to class 1: P(Y=1|X=x), where x is a datapoint observation

#Fix test data to conform to standard input for sklearn, arrays
test11 = test11.rename(columns = {'0':'X1test',
                        '1':'X2test',
                        '2':'y1test'})
Ty1, TX1 = dmatrices('y1test ~ X1test + X2test', test11, return_type="dataframe")
Ty1=np.ravel(Ty1)
TX1 = TX1.reset_index().values
TX1 = np.delete(TX1, [0,1], axis=1)

test22 = test22.rename(columns = {'0':'X1test',
                        '1':'X2test',
                        '2':'y1test'})
Ty2, TX2 = dmatrices('y1test ~ X1test + X2test', test22, return_type="dataframe")
Ty2=np.ravel(Ty1)
TX2 = TX2.reset_index().values
TX2 = np.delete(TX2, [0,1], axis=1)

test33 = test33.rename(columns = {'0':'X1test',
                        '1':'X2test',
                        '2':'y1test'})
Ty3, TX3 = dmatrices('y1test ~ X1test + X2test', test33, return_type="dataframe")
Ty3=np.ravel(Ty3)
TX3 = TX3.reset_index().values
TX3 = np.delete(TX3, [0,1], axis=1)

#LOG prediction prob
log1Prob = log1.predict_proba(TX1)
log2Prob = log2.predict_proba(TX2)
log3Prob = log3.predict_proba(TX3)

#LDA prediction prob
lda1Prob = lda1.predict_proba(TX1)
lda2Prob = lda2.predict_proba(TX2)
lda3Prob = lda3.predict_proba(TX3)

#QDA prediction prob
qda1Prob = qda1.predict_proba(TX1)
qda2Prob = qda2.predict_proba(TX2)
qda3Prob = qda3.predict_proba(TX3)

#build a function getROCdata, that returns a dataframe with the 11 columns listed
#truthvals is a column vector containing the correct classification
#probs is a column vector of probability that the model believes the datapoint to be of class 1
#thresholds is a vector of probability thresholds to use when deciding to predict what class it is 

def getROCdata(truthVals, probs, thresholds): 
    row_array = np.zeros([len(thresholds),11])
    
    for j in range(len(thresholds)):
        add_array = np.zeros([1,11])
        Predict = np.zeros(len(probs))
        TRUPOScount = np.zeros(1)
        TRUNEGcount = np.zeros(1)
        FALPOScount = np.zeros(1)
        FALNEGcount = np.zeros(1)
        for i in range(len(probs)):
            
            if probs[i,1] > thresholds[j]:
                Predict[i] = 1
            #tru positive and true negative
            #tru pos if actual = predicted and val = 1
            #tru neg if actual = predicted and val = 0
            if (Predict[i] == Ty1[i]) & (Ty1[i] == 1):
                #tru positive = predicted pos and pos
                TRUPOScount = TRUPOScount + 1
                
            elif (Predict[i] == Ty1[i]) & (Ty1[i] == 0):
                #true neg = predict neg and neg
                TRUNEGcount = TRUNEGcount + 1
                
            #false positive and false negative
            #false pos if actual != predicted and val = 1
            #false neg if actual != predicted and val = 0 
            if (Predict[i] != Ty1[i]) & (Ty1[i] == 0):
                #false pos = predicted positive and neg
                FALPOScount = FALPOScount + 1
                
            elif (Predict[i] != Ty1[i]) & (Ty1[i] == 1):
                #false neg = predicted neg and pos
                FALNEGcount = FALNEGcount + 1
                
            #needed value calcs for tru positive, false pos, accuracy, precision, recall, F measure
            TRUPOSrate = (TRUPOScount / (TRUPOScount + FALNEGcount))
            FALPOSrate = (FALPOScount / (TRUNEGcount + FALPOScount))
            Accuracy = (TRUNEGcount + TRUPOScount) / len(Predict)
            Precision = (TRUPOScount / (FALPOScount + TRUPOScount))
            Recall = TRUPOSrate
            F_Measure = (2 * Precision * Recall)/(Precision + Recall)
            
            add_array[0,0] =  thresholds[j]
            add_array[0,1] =  TRUPOScount
            add_array[0,2] =  FALPOScount
            add_array[0,3] =  TRUNEGcount 
            add_array[0,4] =  FALNEGcount 
            add_array[0,5] =  TRUPOSrate 
            add_array[0,6] =  FALPOSrate 
            add_array[0,7] =  Accuracy
            add_array[0,8] =  Precision
            add_array[0,9] =  Recall
            add_array[0,10]=  F_Measure
                      
        row_array[j] = add_array
        
    ROCdf = pd.DataFrame(row_array, columns =['Threshold', 'True_Positive_Count', 'False_Positive_Count',
                                              'True_Negative_Count', 'False_Negative_Count', 'True_Positive_Rate',
                                              'False_Positive_Rate', 'Accuracy', 'Precision', 'Recall',
                                              'F_Measure'])
    
    return ROCdf
        
#8. for each model, smartly generate a vector of 100 probability thresholds to test and call
#retROCdata function and obtain a response
lower1log = np.amin(log1Prob) 
lower1lda = np.amin(lda1Prob) 
lower1qda = np.amin(qda1Prob)
upper1log = np.amax(log1Prob) 
upper1lda = np.amax(lda1Prob) 
upper1qda = np.amax(qda1Prob)
thresholds1log = np.linspace(lower1log,upper1log,num=100)
thresholds1lda = np.linspace(lower1lda,upper1lda,num=100)
thresholds1qda = np.linspace(lower1qda,upper1qda,num=100)

lower2log = np.amin(log2Prob) 
lower2lda = np.amin(lda2Prob) 
lower2qda = np.amin(qda2Prob)
upper2log = np.amax(log2Prob) 
upper2lda = np.amax(lda2Prob) 
upper2qda = np.amax(qda2Prob)
thresholds2log = np.linspace(lower2log,upper2log,num=100)
thresholds2lda = np.linspace(lower2lda,upper2lda,num=100)
thresholds2qda = np.linspace(lower2qda,upper2qda,num=100)

lower3log = np.amin(log3Prob) 
lower3lda = np.amin(lda3Prob) 
lower3qda = np.amin(qda3Prob)
upper3log = np.amax(log3Prob) 
upper3lda = np.amax(lda3Prob) 
upper3qda = np.amax(qda3Prob)
thresholds3log = np.linspace(lower3log,upper3log,num=100)
thresholds3lda = np.linspace(lower3lda,upper3lda,num=100)
thresholds3qda = np.linspace(lower3qda,upper3qda,num=100)

#LOG regression ROC data, all three sets
log1ROC = getROCdata(Ty1, log1Prob, thresholds1log)
log2ROC = getROCdata(Ty2, log2Prob, thresholds2log)
log3ROC = getROCdata(Ty3, log3Prob, thresholds3log)
#LDA ROC data, all three sets
lda1ROC = getROCdata(Ty1, lda1Prob, thresholds1lda)
lda2ROC = getROCdata(Ty2, lda2Prob, thresholds2lda)
lda3ROC = getROCdata(Ty3, lda3Prob, thresholds3lda)
#QDA ROC data, all three sets
qda1ROC = getROCdata(Ty1, qda1Prob, thresholds1qda)
qda2ROC = getROCdata(Ty2, qda2Prob, thresholds2qda)
qda3ROC = getROCdata(Ty3, qda3Prob, thresholds3qda)

#9. using the reponse from getROCdata function, plot the ROC curves for each of the three classifiers
#on a single plot
#X axis is the false positive rate
#Y axis is the true positive rate

#dataset 1
logROCx1 = log1ROC.False_Positive_Rate
logROCy1 = log1ROC.True_Positive_Rate

ldaROCx1 = lda1ROC.False_Positive_Rate
ldaROCy1 = lda1ROC.True_Positive_Rate

qdaROCx1 = qda1ROC.False_Positive_Rate
qdaROCy1 = qda1ROC.True_Positive_Rate
names = ['LOG Regression', 'LDA', 'QDA']

plt.plot(logROCx1, logROCy1, 'r')
plt.plot(ldaROCx1, ldaROCy1, 'g')
plt.plot(qdaROCx1, qdaROCy1, 'b')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(names)
plt.title('ROC Curve Dataset 1')
plt.show()

#dataset2
logROCx2 = log2ROC.False_Positive_Rate
logROCy2 = log2ROC.True_Positive_Rate

ldaROCx2 = lda2ROC.False_Positive_Rate
ldaROCy2 = lda2ROC.True_Positive_Rate
        
qdaROCx2 = qda2ROC.False_Positive_Rate
qdaROCy2 = qda2ROC.True_Positive_Rate

plt.plot(logROCx2, logROCy2, 'r')
plt.plot(ldaROCx2, ldaROCy2, 'g')
plt.plot(qdaROCx2, qdaROCy2, 'b')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(names)
plt.title('ROC Curve Dataset 2')
plt.show()

#dataset3
logROCx3 = log3ROC.False_Positive_Rate
logROCy3 = log3ROC.True_Positive_Rate

ldaROCx3 = lda3ROC.False_Positive_Rate
ldaROCy3 = lda3ROC.True_Positive_Rate
        
qdaROCx3 = qda3ROC.False_Positive_Rate
qdaROCy3 = qda3ROC.True_Positive_Rate

plt.plot(logROCx3, logROCy3, 'r')
plt.plot(ldaROCx3, ldaROCy3, 'g')
plt.plot(qdaROCx3, qdaROCy3, 'b')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(names)
plt.title('ROC Curve Dataset 3')
plt.show()

#10. Develop code to compute the area under the curve for ROC curves for each of the three classifiers
#you will need to deal with partial inforamtion, since not all of the curves may extend
# from 0 to 1
#with some help from stack exchange, we can calculate the area under a trapezoid
#and "integrate" using trapezoids
#this will allow us to cutoff the limit of integtration at the greatest value 
#within the range of the ROC curve, rather than 0 to 1

#assumptions
#since we can generate the threshold values looking at the data and determining
#some optimal range, we start that way...
#this however, led to partial ROC curves, that did not give a good estimate of AUC
#so, we then calculated the minimum values from the probabilities and the max values
#and used this to generate the threshold, which results in some tail degeneracy,
#however it allowed the graph to project from 0 to 1, thus allowing a total
#computation for AUC
#we also had to take the absolute value produced from np.trapz, based on the way values
#for thresholds were generated.  this resulted in a negative AUC, but we know that
#based on the decreasing values in the ROC curve, the inverse is true, 
#a positive area under the curve

#compute AUC for logistic regression
log1AUC = abs(np.trapz(logROCy1, logROCx1))
log2AUC = abs(np.trapz(logROCy2, logROCx2))
log3AUC = abs(np.trapz(logROCy3, logROCx3))
print('LOG Area Under Curve')
print('Data Set 1')
print(log1AUC)
print('Data Set 2')
print(log2AUC)
print('Data Set 3')
print(log3AUC)

#LDA AUC
lda1AUC = abs(np.trapz(ldaROCy1, ldaROCx1))
lda2AUC = abs(np.trapz(ldaROCy2, ldaROCx2))
lda3AUC = abs(np.trapz(ldaROCy3, ldaROCx3))
print('LDA Area Under Curve')
print('Data Set 1')
print(lda1AUC)
print('Data Set 2')
print(lda2AUC)
print('Data Set 3')
print(lda3AUC)

#QDA AUC
qda1AUC = abs(np.trapz(qdaROCy1, qdaROCx1))
qda2AUC = abs(np.trapz(qdaROCy2, qdaROCx2))
qda3AUC = abs(np.trapz(qdaROCy3, qdaROCx3))
print('QDA Area Under Curve')
print('Data Set 1')
print(qda1AUC)
print('Data Set 2')
print(qda2AUC)
print('Data Set 3')
print(qda3AUC)

#11.  Using the ROCdata, for each model, determine the probability threshold(s) for which the
#following performance measures are maximized:
# Accuracy, Precision, Recall, F-Measure.  Report a confusion matrix of predicted v true class at 
#each threshold value.  examing them, explain what tradeoff is occuring when we set probability threshold differently
#to max eaech of those measures

log1ROC.ix[log1ROC['Accuracy'].idxmax(), 'Threshold']



def ConfMat (ROCdata):
    #take in ROC data
    #pull out needed data
    #put into new matrix
    ConMatrixACC = {'Predict 0':pd.Series([ROCdata.ix[ROCdata['Accuracy'].idxmax(), 'True_Negative_Count'],
                ROCdata.ix[ROCdata['Accuracy'].idxmax(), 'False_Negative_Count']], index = ['Actual 0', 'Actual 1']),
                'Predict 1': pd.Series([ROCdata.ix[ROCdata['Accuracy'].idxmax(), 'False_Positive_Count'],
                ROCdata.ix[ROCdata['Accuracy'].idxmax(), 'True_Positive_Count']], index = ['Actual 0' ,'Actual 1'])}
    ConMatrixACC = pd.DataFrame(ConMatrixACC)
    
    ConMatrixPre = {'Predict 0':pd.Series([ROCdata.ix[ROCdata['Precision'].idxmax(), 'True_Negative_Count'],
                ROCdata.ix[ROCdata['Precision'].idxmax(), 'False_Negative_Count']], index = ['Actual 0', 'Actual 1']),
                'Predict 1': pd.Series([ROCdata.ix[ROCdata['Precision'].idxmax(), 'False_Positive_Count'],
                ROCdata.ix[ROCdata['Precision'].idxmax(), 'True_Positive_Count']], index = ['Actual 0' ,'Actual 1'])}
    ConMatrixPre = pd.DataFrame(ConMatrixPre)
    
    ConMatrixRec = {'Predict 0':pd.Series([ROCdata.ix[ROCdata['Recall'].idxmax(), 'True_Negative_Count'],
                ROCdata.ix[ROCdata['Recall'].idxmax(), 'False_Negative_Count']], index = ['Actual 0', 'Actual 1']),
                'Predict 1': pd.Series([ROCdata.ix[ROCdata['Recall'].idxmax(), 'False_Positive_Count'],
                ROCdata.ix[ROCdata['Recall'].idxmax(), 'True_Positive_Count']], index = ['Actual 0' ,'Actual 1'])}
    ConMatrixRec = pd.DataFrame(ConMatrixRec)
    
    ConMatrixF = {'Predict 0':pd.Series([ROCdata.ix[ROCdata['F_Measure'].idxmax(), 'True_Negative_Count'],
                ROCdata.ix[ROCdata['F_Measure'].idxmax(), 'False_Negative_Count']], index = ['Actual 0', 'Actual 1']),
                'Predict 1': pd.Series([ROCdata.ix[ROCdata['F_Measure'].idxmax(), 'False_Positive_Count'],
                ROCdata.ix[ROCdata['F_Measure'].idxmax(), 'True_Positive_Count']], index = ['Actual 0' ,'Actual 1'])}
    ConMatrixF = pd.DataFrame(ConMatrixF)
    return ConMatrixACC, ConMatrixPre, ConMatrixRec, ConMatrixF

[lda1ACCMat, lda1PreMat,lda1RecMat,lda1FMat] = ConfMat(lda1ROC)
[lda1ACCMat, lda1PreMat,lda1RecMat,lda1FMat] = ConfMat(lda2ROC)
[lda1ACCMat, lda1PreMat,lda1RecMat,lda1FMat] = ConfMat(lda3ROC)

[log1ACCMat, log1PreMat,log1RecMat,log1FMat] = ConfMat(log1ROC)
[log2ACCMat, log2PreMat,log2RecMat,log2FMat] = ConfMat(log2ROC)
[log3ACCMat, log3PreMat,log3RecMat,log3FMat] = ConfMat(log3ROC)

[qda1ACCMat, qda1PreMat,qda1RecMat,qda1FMat] = ConfMat(qda1ROC)
[qda2ACCMat, qda2PreMat,qda2RecMat,qda2FMat] = ConfMat(qda2ROC)
[qda3ACCMat, qda3PreMat,qda3RecMat,qda3FMat] = ConfMat(qda3ROC)

#12. Add text or points to the ROC graph which represent the max accuracy, precision, recall and F measure.  
#what do you notice about these points?

def plotPoints(ROCdata):

    ACCptx = ROCdata.ix[ROCdata['Accuracy'].idxmax(), 'False_Positive_Rate']
    ACCpty = ROCdata.ix[ROCdata['Accuracy'].idxmax(), 'True_Positive_Rate']
    
    
    PrePtx = ROCdata.ix[ROCdata['Precision'].idxmax(), 'False_Positive_Rate']
    PrePty = ROCdata.ix[ROCdata['Precision'].idxmax(), 'True_Positive_Rate']
    
    RecPtx = ROCdata.ix[ROCdata['Recall'].idxmax(), 'False_Positive_Rate']
    RecPty = ROCdata.ix[ROCdata['Recall'].idxmax(), 'True_Positive_Rate']
    
    Fptx = ROCdata.ix[ROCdata['F_Measure'].idxmax(), 'False_Positive_Rate']
    Fpty = ROCdata.ix[ROCdata['F_Measure'].idxmax(), 'True_Positive_Rate']
    
    return ACCptx, ACCpty, PrePtx, PrePty, RecPtx, RecPty, Fptx, Fpty

log1ACCptx, log1ACCpty, log1PrePtx, log1PrePty, log1RecPtx, log1RecPty, log1Fptx, log1Fpty = plotPoints(log1ROC)
log2ACCptx, log2ACCpty, log2PrePtx, log2PrePty, log2RecPtx, log2RecPty, log2Fptx, log2Fpty = plotPoints(log2ROC)
log3ACCptx, log3ACCpty, log3PrePtx, log3PrePty, log3RecPtx, log3RecPty, log3Fptx, log3Fpty = plotPoints(log3ROC)

lda1ACCptx, lda1ACCpty, lda1PrePtx, lda1PrePty, lda1RecPtx, lda1RecPty, lda1Fptx, lda1Fpty = plotPoints(lda1ROC)
lda2ACCptx, lda2ACCpty, lda2PrePtx, lda2PrePty, lda2RecPtx, lda2RecPty, lda2Fptx, lda2Fpty = plotPoints(lda2ROC)
lda3ACCptx, lda3ACCpty, lda3PrePtx, lda3PrePty, lda3RecPtx, lda3RecPty, lda3Fptx, lda3Fpty = plotPoints(lda3ROC)

qda1ACCptx, qda1ACCpty, qda1PrePtx, qda1PrePty, qda1RecPtx, qda1RecPty, qda1Fptx, qda1Fpty = plotPoints(qda1ROC)
qda2ACCptx, qda2ACCpty, qda2PrePtx, qda2PrePty, qda2RecPtx, qda2RecPty, qda2Fptx, qda2Fpty = plotPoints(qda2ROC)
qda3ACCptx, qda3ACCpty, qda3PrePtx, qda3PrePty, qda3RecPtx, qda3RecPty, qda3Fptx, qda3Fpty = plotPoints(qda3ROC)

names = ['LOG Regression', 'LDA', 'QDA', 'MAX Accuracy', 'MAX Precision', 'MAX Recall', 'MAX F-Measure']
#dataset 1
plt.plot(logROCx1, logROCy1, 'r')
plt.plot(ldaROCx1, ldaROCy1, 'g')
plt.plot(qdaROCx1, qdaROCy1, 'b')
plt.plot([log1ACCptx, lda1ACCptx, qda1ACCptx], [log1ACCpty,lda1ACCpty,qda1ACCpty], 'kp')
plt.plot([log1PrePtx, lda1PrePtx, qda1PrePtx], [log1PrePty,lda1PrePty,qda1PrePty], 'co')
plt.plot([log1RecPtx, lda1RecPtx, qda1RecPtx], [log1RecPty,lda1RecPty,qda1RecPty], 'm8')
plt.plot([log1Fptx, lda1Fptx, qda1Fptx], [log1Fpty,lda1Fpty,qda1Fpty], 'yX')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(names)
plt.title('ROC Curve Dataset 1')
plt.show()

#dataset2
plt.plot(logROCx2, logROCy2, 'r')
plt.plot(ldaROCx2, ldaROCy2, 'g')
plt.plot(qdaROCx2, qdaROCy2, 'b')
plt.plot([log2ACCptx, lda2ACCptx, qda2ACCptx], [log2ACCpty,lda2ACCpty,qda2ACCpty], 'kp')
plt.plot([log2PrePtx, lda2PrePtx, qda2PrePtx], [log2PrePty,lda2PrePty,qda2PrePty], 'co')
plt.plot([log2RecPtx, lda2RecPtx, qda2RecPtx], [log2RecPty,lda2RecPty,qda2RecPty], 'm8')
plt.plot([log2Fptx, lda2Fptx, qda2Fptx], [log2Fpty,lda2Fpty,qda2Fpty], 'yX')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(names)
plt.title('ROC Curve Dataset 2')
plt.show()

#dataset3
plt.plot(logROCx3, logROCy3, 'r')
plt.plot(ldaROCx3, ldaROCy3, 'g')
plt.plot(qdaROCx3, qdaROCy3, 'b')
plt.plot([log3ACCptx, lda3ACCptx, qda3ACCptx], [log3ACCpty,lda3ACCpty,qda3ACCpty], 'kp')
plt.plot([log3PrePtx, lda3PrePtx, qda3PrePtx], [log3PrePty,lda3PrePty,qda3PrePty], 'co')
plt.plot([log3RecPtx, lda3RecPtx, qda3RecPtx], [log3RecPty,lda3RecPty,qda3RecPty], 'm8')
plt.plot([log3Fptx, lda3Fptx, qda3Fptx], [log3Fpty,lda3Fpty,qda3Fpty], 'yX')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(names)
plt.title('ROC Curve Dataset 3')
plt.show()