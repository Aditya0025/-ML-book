# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 00:52:09 2018

@author: adity
"""

from sklearn.datasets import load_breast_cancer # loading data sets
from sklearn.linear_model import LogisticRegression # machine learningmodel
from sklearn.model_selection import train_test_split # for data spliliting

import matplotlib.pyplot as plt
import mglearn
mglearn.plots.plot_linear_regression_wave()


cancer = load_breast_cancer()
print(cancer.data)
X_train, X_test, Y_train, Y_test =  train_test_split(cancer.data, cancer.target, stratify=cancer.target,
                                                     random_state= 42)# dta= features and targets are labels
clf = LogisticRegression()
clf.fit(X_train, Y_train) 
print("accuacy score of trainf set: {:.3f}".format(clf.score(X_train,Y_train)))
print("accuracy score of tesst data: {:.3f}".format(clf.score(X_test,Y_test)))

"""can be play with the parameter for better results

n more specific terms, we can think of regularization as adding (or increasing the)
 bias if our model suffers from (high) variance (i.e., it overfits the training data). 
 On the other hand, too much bias will result in underfitting (a characteristic indicator
 of high bias is that the model shows a "bad" performance for both the training and test dataset).
 We know that our goal in an unregularized model is to minimize the cost function, 
 i.e., we want to find the feature weights that correspond to the global cost minimum
 (remember that the logistic cost function is convex).

logistic regrestion use regulaorization to avoid over fitting
L1 = assume only a few features are important 
L2= but not assume only q few feature are important - used by defsult in scikit-learn

'C' parametre to control the strength of regularization(conrol the s-accuracy at our data set)
lower c = log_reg adjust to the majority of datapoints
upper c = correct classification of each data point

by default c=1
"""
clf2 = LogisticRegression(C=100)
clf2.fit(X_train, Y_train)

print("accuracy of train data: {:.3f}".format(clf2.score(X_train,Y_train)))
print("accuracy of test data: {:.3f}".format(clf2.score(X_test,Y_test)))



clf3 = LogisticRegression(C=0.01)
clf3.fit(X_train,Y_train)
print("accuarcy of the train data: {:.3f}".format(clf3.score(X_train,Y_train)))
print("accuarcy of the test data: {:.3f}".format(clf3.score(X_test,Y_test)))



plt.plot(clf.coef_.T,'^', label= "clf c=1") #coef_t
plt.plot(clf2.coef_.T,'^', label= "clf2 c=100")
plt.plot(clf3.coef_.T,'v',label= "clf3 c=0.01")
# simple to ticks for gap btw point(intervals) and puting mane to them
plt.xticks(range(cancer.data.shape[1]),cancer.feature_names, rotation=90)

plt.hlines(0,0, cancer.data.shape[1])
plt.ylim(5,-5)
plt.xlabel("cofficent index")
plt.ylabel('coffient of magnitude')
plt.legend()













