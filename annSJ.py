# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:46:10 2018

@author: GTX
"""
# Importing the Libraries
import numpy as py
import matplotlib.pyplot as plt
import pandas as pd


# PART - 1 Data PreProcessing
# Importing Dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values
X

# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_1.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

# Feature Scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# PART - 2 Making ANN
# Importing the Keras Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the Input layer and the First hidden layer
classifier.add(Dense( units = 6, activation="relu", kernel_initializer="uniform", input_dim = 11))

# Adding the Second hidden layer
classifier.add(Dense( units = 6, activation="relu", kernel_initializer="uniform"))

# Adding the output layer
classifier.add(Dense( units = 1, activation="sigmoid", kernel_initializer="uniform"))

# Compiling the ANN
classifier.compile(optimizer =  "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Fitting the ANN to the Training Set
classifier.fit(X_train, y_train, batch_size =10, epochs =100)

# PART - 3 Making the prediction and evaluating the model
# Predicting the Test Set results
y_pred = classifier.predict(X_test)
y_pred = y_pred > 0.5

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Accuracy
(cm[0][0] + cm[1][1])/(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
