
from sklearn.datasets import load_breast_cancer # loading data
from sklearn.neighbors import KNeighborsClassifier # machine learning algoritm
from sklearn.model_selection import train_test_split #load the train dataset

import matplotlib.pyplot as plt
import mglearn

cancer = load_breast_cancer()
#print(cancer.DESCR)
#mglearn.plots.plot_knn_classification(n_neighbors=5)

X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, 
                                                    stratify =cancer.target, random_state =42  )
# startify useed to data split and shoe to which fashioned it should spilit
"""data comes in to different type trainig data and testing data"""

"""print('X_train' , X_train)# feature
print('X_test',X_test)#label
print('Y_train', Y_train)
print('Y_test',Y_test)"""

knn = KNeighborsClassifier()
# by default it will search 5 neares neighbour
knn.fit(X_train,Y_train)

#evaluation
print("accuracy of knn n-5 , on taining set: {:.3f}".format(knn.score(X_train,Y_train)))
print("acuray of knn n-5 , on test data set {:.3f}" .format(knn.score(X_test,Y_test)))