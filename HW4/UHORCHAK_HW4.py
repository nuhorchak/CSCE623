# -*- coding: utf-8 -*-
"""
Created on Sun May  7 20:06:32 2017

@author: nuhorchak
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import seaborn as sns
from itertools import combinations
import time

# 1.	Using pandas, load the “ds_10.csv” dataset which contains a synthetic dataset with a response variable
# Y and several features/predictors p1…p10).  Note that the dataset is sorted such that all the class 1 points #
#eare in the first 150 rows and all the class 2 points are in the remaining 150 rows (you will need to shuffle 
#the data before splitting it into training, validation, and testing sets (50 observations from each class in each 
#set).  
#Hint – use train_test_split twice.

state = 1
data = pd.read_csv('ds_10.csv')
data = data.drop(['Unnamed: 0'], axis=1)
data = data.sample(frac=1,random_state=state)

def Test_Train(data, split):

    #ensure even distribution by splitting by 0 and 1
    upper_data = data[data['Y'] == 1]
    lower_data = data[data['Y'] == 2]
    #generate random split of 0 and 1 data, by specified fraction
    train_Upper = upper_data.sample(frac=(split/len(data)),random_state=state)
    test_Upper = upper_data.drop(train_Upper.index)
    train_Lower = lower_data.sample(frac=split/len(data),random_state=state)
    test_Lower = lower_data.drop(train_Lower.index)
    #final merge of uniform splt of data
    train_df = [train_Upper,train_Lower]
    test_df = [test_Upper,test_Lower]
    train = pd.concat(train_df)
    test = pd.concat(test_df)  
    return train, test


Test, Train = Test_Train(data,100)
Train, Validate = Test_Train(Train, 100)


#2.	Explore the data using techniques from class and previous homework.  
#Your goal for this exploration step is to try to determine (with your eyeballs) salient features that you 
#think will make good features/predictors for a Linear classification model such as Logistic Regression or LDA. 


#x_var = [data['p1'], data['p2'], data['p3'], data['p4'], data['p5'], data['p6'], data['p7'], data['p8'], data['p9'], data['p10']]
#sns.set()
#sns.pairplot(data, hue="Y", x_vars=x_var, y_vars=data['Y'])

#4.	Build a function bestSubset(train, val, k) to implement part of algorithm 6.1 (page 205): 
#steps 1 and 2.  The training and validation datasets should be in the form of pandas 
#dataframes with column headers indicating feature identifiers and the class label “Y”.  
#Your function should return both the validation set classification accuracy and the best set
#of k features found for the model – which are the dataframe feature column headers.  
#Best is defined as the model which, when trained on the training data, 
#has the best classification accuracy on the validation data.  Use LDA for your classifier.  
#This may take a while – for a model of size k you will need to fit and evaluate 2k models. 
#Note – I want you to design the code for evaluating the best subset of features yourself – 
#don’t use some python package to determine best subset.

def bestSubset(train, val, k):
    #tic = time.time()
    #train
    y_train = train['Y']
    x_data = train
    x_data = x_data.drop(['Y'],axis=1)
        #validate
    y_valid = val['Y']
    x_valid = val
    x_valid = x_valid.drop(['Y'],axis=1)
    
    scores = []
    #num_features = x_data.shape[1]
    subsets = list(combinations(x_data.columns,k))
    
    for subset in subsets:
        lda = LDA()
        lda.fit(x_data[list(subset)], y_train)
        scores.append(lda.score(x_valid[list(subset)], y_valid))
    max_score = np.amax(scores)
    max_ind = scores.index(max_score)
    best_set = subsets[max_ind]
    #toc = time.time()
    #print(toc-tic)
    return max_score, best_set

#5.	Execute the bestSubset() function for model size values that range from 1 to p to obtain the p best
# sets of features (1 set for each model size).  Print out a table of the best features per model size (k)
# – for example, like the output shown in the lab on page 245.   Discuss any interesting changes in what
# the model chooses as features – for instance, did a feature which was selected when k = 3 not get selected
# when k > 3?  If so,  explain why?

best_sets = []
max_scores = []
tic = time.clock()
for i in range(1,11): 
    score, sets = bestSubset(Train,Validate,i)
    best_sets.append(sets)
    max_scores.append(score)
toc = time.clock()
print('Best Subset total Run Time:', toc-tic, 'seconds')
    
bests_df = pd.DataFrame({'Best Subsets': best_sets, 'Max Score':max_scores})
print(bests_df)

#6.	Create a plot of the validation set classification accuracy of each of the p best models 
#(as returned from bestSubset) vs. k.   Describe the change in these values as the model size grows 
#from 1 to 10. 

for i in range(0,10):
    plt.plot(i+1,max_scores[i],'go')
    
plt.xlabel('k')
plt.ylabel('Validation Score')
plt.title('Accuracy vs K features')
plt.xticks([1,2,3,4,5,6,7,8,9,10])

#7.	Annotate your plot created in step 6 with the point that yields the best performing model
# (that maximize or minimize the performance estimation criteria you plotted). 
# This point reveals the best k.

best_k = np.amax(max_scores)
best_k_ind = max_scores.index(best_k)+1
x = np.arange(1,11)
plt.figure()
for i in range(0,10):
    plt.plot(i+1,max_scores[i],'go')
plt.plot(x,max_scores)
    
plt.xlabel('k')
plt.ylabel('Validation Score')
plt.title('Accuracy vs K features')
plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.scatter(best_k_ind, best_k,s=200, facecolors='none', edgecolors='r', label="Best K")
plt.legend(loc='lower right')
plt.show()

#8.	Report the validation set classification accuracy on the model with the best k features.
print("Best subset", best_k)

#10.	Write a function forwardStepwiseSubset(train, val, q)to perform forward stepwise selection on
# a dataset as shown in algorithm 6.2 (page 207) steps 1 and 2.  Your function should return both the 
#validation set classification accuracy and the step-wise-selected set of q features found for the model – 
#which are the dataframe feature column headers. You should design your code such that in step 2(b) you 
#fit LDA for each model using the k features currently under consideration.    The “step-wise-best” model
# here is the one with the highest classification accuracy on the validation set.    

def forwardStepwiseSubset(train, val, q):
    #tic = time.time()
    #train
    y_train = train['Y']
    x_data = train
    x_data = x_data.drop(['Y'],axis=1)
    #validate
    y_valid = val['Y']
    x_valid = val
    x_valid = x_valid.drop(['Y'],axis=1)
    
    max_score = []
    scores = []
    predictors = []

    col_names = list(x_data.columns)
    for i in range(1,q+1):
        
        remaining_predictors = [p for p in col_names if p not in predictors]
        scores = []
        for p in remaining_predictors:
            
            lda = LDA()
            lda.fit(x_data[(predictors)+[p]], y_train)
            scores.append(lda.score(x_valid[(list(predictors)+[p])], y_valid))
            
        max_score = np.amax(scores)
        max_ind = scores.index(max_score)
        predictors = predictors + [remaining_predictors[max_ind]]
    #toc = time.time()
    #print(toc-tic)
    return max_score, predictors

#11.	Execute the forwardStepwiseSubset() function for model size q values that range
# from 1 to p to obtain the p best stepwise-generated sets of features (1 set for each model size).
#  Print out a table of the best features per model size (q) – for example, like the output shown 
#in the lab on page 245.   Discuss how the stepwise-selected features changed compared to how the 
#best-selected features changed (Part 2, step 5)

best_FWDsets = []
max_FWDscores = []
tic1 = time.clock()
for i in range(1,11):    
    score, sets = forwardStepwiseSubset(Train,Validate,i)
    best_FWDsets.append(sets)
    max_FWDscores.append(score)
toc1 = time.clock()
print('Forward subset total Run Time:', toc1-tic1, 'seconds')
bestFWD_df = pd.DataFrame({'Best Subsets': best_FWDsets, 'Max Score':max_FWDscores})
bestFWD_df

print(bestFWD_df)

#12.	Add a different color line to your plot from step 6:   
    #plot the validation set classification accuracy of each of the p best models%re
    #(as returned from forwardStepwiseSubset) vs. q.   Describe the change in these 
    #values as the model size grows from 1 to 10. 


best_k = np.amax(max_scores)
best_k_ind = max_scores.index(best_k)+1
best_kFWD = np.amax(max_FWDscores)
best_kFWD_ind = max_FWDscores.index(best_kFWD)+1
                             
x = np.arange(1,11)
plt.figure()
plt.plot(x,max_scores)
plt.plot(x,max_FWDscores)
    
plt.xlabel('k')
plt.ylabel('Validation Score - Best Subset and Foward subset')
plt.title('Accuracy vs K features')
plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.scatter(best_k_ind, best_k,s=200, facecolors='none', edgecolors='r', label="Best K")
plt.scatter(best_kFWD_ind, best_kFWD,s=400, facecolors='none', edgecolors='g', label="Best K FWD")
plt.legend(loc='lower right')
plt.show()

#13.	Annotate your plot created in step 12 with the point that yields the stepwise 
#best performing model (that maximize classification accuracy you plotted).  
#This point reveals the best q. Report the classification accuracy on the 
#validation set for this best model.
print("Best Forward",best_kFWD)

#14.	Using the best validation accuracy feature sets from each algorithm 
#(best subset; forward stepwise), fit a LDA model on the combined training+validation data.  
#Then use each of two fit models to compute the LDA classification accuracy on the test set.  
#Report the results.   Did the optimal subset best model beat the greedy subset model?
#train val sets combined
Train_Val = pd.concat([Train, Validate])
Train_Val = Train_Val.reset_index()
Train_Val = Train_Val.drop('index', axis=1)
y_Train_Val = Train_Val['Y']
#best indices for building dataframe from index values
best_set_subset = list(best_sets[best_k_ind])
best_set_FWD = list(best_FWDsets[best_kFWD_ind])
#build best test and best test forward DF's for LDA
best_train = Train_Val.loc[:,best_set_subset]
FWD_train = Train_Val.loc[:,best_set_FWD]

#test sets
y_test_best = Test['Y']
x_test_best = Test.loc[:,best_set_subset]

y_test_FWD = Test['Y']
x_test_FWD = Test.loc[:,best_set_FWD]

#best subset model
lda = LDA()
lda.fit(best_train, y_Train_Val)
scores_Best = lda.score(x_test_best, y_test_best)
#forward subset model

lda = LDA()
lda.fit(FWD_train, y_Train_Val)
scores_FWD = lda.score(x_test_FWD, y_test_FWD)

print('Best subset score:', scores_Best)
print('Forward subset score:', scores_FWD)
