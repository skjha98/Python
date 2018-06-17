# Creating own Classifier KNN
import random
class SexyClassifier():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        prediction = []
        for row in X_test:
            label = random.choice(self.y_train)
            prediction.append(label)
        return prediction



# Importing Libraries
from sklearn.datasets import load_iris
import numpy as np
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Importing the iris datasets
dataset = load_iris()

# train test split
data_train, data_test, target_train, target_test = train_test_split(dataset.data, dataset.target, test_size = 0.2)

# Training the classifier
clf = SexyClassifier()
clf.fit(data_train, target_train)

# Predicting the test results
predict = clf.predict(data_test)
print(predict, target_test)

# Generating the accuracy_score
print(accuracy_score(predict,target_test))
