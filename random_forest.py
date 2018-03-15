# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 15:24:26 2018

@author: adity
"""
"""" working on samples"""
""" randomforest can be think as a collection of decision trees
after finding the each tree average is done after thwt
estimator are used to estimate hoe many tree will be made
""" 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

import matplotlib.pyplot as plt
import numpy as np

cancer = load_breast_cancer()

X_train,X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target ,
                                                   stratify= cancer.target, random_state=0)

clf = RandomForestClassifier(n_estimators= 100, random_state=0)
clf.fit(X_train,Y_train)

print("accuracy : {:.3f}".format(clf.score(X_train,Y_train)))
print("accuracy : {:.3f}".format(clf.score(X_test,Y_test)))

n_feature = cancer.data.shape[1]
plt.barh(range(n_feature), clf.feature_importances_, align='center')
plt.yticks(np.arange(n_feature), cancer.feature_names)
plt.xlabel("feature impotrancec")
plt.ylabel("feature")
plt.show()















 