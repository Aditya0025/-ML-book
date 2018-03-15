
from sklearn.datasets import load_breast_cancer # loading data
from sklearn.neighbors import KNeighborsClassifier # machine learning algoritm
from sklearn.model_selection import train_test_split #load the train dataset

import matplotlib.pyplot as plt
import mglearn

cancer = load_breast_cancer()
#print(cancer.DESCR)
#mglearn.plots.plot_knn_classification(n_neighbors=5)

#  X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, 
#                                                    stratify =cancer.target, random_state =42  )
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

# try when we decrease the no of neighbours 
"""for this we try the no of neghbr from 1-10 for this we use a loop"""

X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target,
                                     ,czx                 stratify= cancer.target, random_state=66)
 
train_accuracy = []
test_accuracy = []
 
neighbour_setting = range(1,11)
 
for n_neighbour in neighbour_setting:
    
    
    clf= KNeighborsClassifier(n_neighbors = n_neighbour)
    clf.fit(X_train,Y_train)
     
    train_accuracy.append(clf.score(X_train,Y_train))
    test_accuracy.append(clf.score(X_test,Y_test))
     
# lets plot them in graph
plt.plot(neighbour_setting, train_accuracy , label= "acccuracy of tarining set")
plt.plot(neighbour_setting, test_accuracy, label="accuracy of test data")
  
plt.xlabel('accuarcy')

plt.ylabel('no_neigh')
plt.legend()
plt.grid(1)
     
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 