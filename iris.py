# Importing Libraries
from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score

# Importing the iris datasets
dataset = load_iris()

# train test split
tdx = [0,15,30,45,60,75,90,105,120,135]
data_train = np.delete(dataset.data, tdx, axis = 0)
target_train = np.delete(dataset.target, tdx, axis = 0)
data_test = dataset.data[tdx]
target_test = dataset.target[tdx]

# Training the classifier
clf = tree.DecisionTreeClassifier()
clf.fit(data_train, target_train)

# Predicting the test results
predict = clf.predict(data_test)
print(predict, target_test)

# Generating the accuracy_score
print(accuracy_score(predict,target_test))
