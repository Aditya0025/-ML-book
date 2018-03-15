# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 01:34:34 2018

@author: adity
"""
""" MLP is generalization of a linear model which basically go through mutiple
process  to select thr possible label or class"""
import graphviz
import mglearn 
mglearn.plots.plot_logistic_regression_graph()

# in MLP weight are calculated multiple times

mglearn.plots.plot_multi_hidden_layer_graph()

#deep neural network
mglearn.plots.plot_two_hidden_layer_graph()

#train the neural net on cancer data

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train,X_test,Y_train,Y_test = train_test_split(cancer.data,cancer.target, random_state=0)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train,Y_train)

print('accuracy on the trainig subset:{:.3f}'.format(mlp.score(X_train,Y_train)))
print('accuracy on the trainig subset:{:.3f}'.format(mlp.score(X_test,Y_test)))

""" scaling the feature puting the values between 0 & 1"""
print('Mximum of each feature in line:\n{}'.format(cancer.data.max(axis=0)))


























 
















