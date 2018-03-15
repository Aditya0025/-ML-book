# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 02:36:39 2018

@author: adity
"""
"""Overfitting is the big issue in decion tree"""
from sklearn.datasets import load_breast_cancer

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, Y_train, Y_test =train_test_split(cancer.data, cancer.target, 
                                                   stratify= cancer.target, random_state=42)  


clf = DecisionTreeClassifier()
clf.fit(X_train,Y_train)

print("score of the clf : {:.3f}".format(clf.score(X_train,Y_train)))
print("score of the clf : {:.3f}".format(clf.score(X_test,Y_test)))

""" overfitting score of train data  due to decision tree is unrestricted
#proning is used to become overfitting problem (pre and post)
# pre pruning stop the creation of the treee at early state by limiing  depth height etc`
POST PRINING REMOVING IRRELEVANT NODES(LITTLE INFO)"""

clf1 = DecisionTreeClassifier(max_depth=4, random_state=0)
clf1.fit(X_train,Y_train)

print("score of the clf : {:.3f}".format(clf1.score(X_train,Y_train)))
print("score of the clf : {:.3f}".format(clf1.score(X_test,Y_test)))

""" install graphwiz for visualization purposes
"""