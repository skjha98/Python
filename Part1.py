# 5 Steps to Go
# Part1

"""
Steps to follow:
1. Load the dataset
2. Preprocess/Augment the model
3. Train the model
4. Test a model
Deploy a model
"""
#Linear Regression
#Load the dataset
from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
dataset = load_boston()
print(dataset.DESCR)
print(dataset.data[5])
print(dataset.target[5])

#Preprocessing the dataset
train_data = np.delete(dataset.data,[1,5,50,100,200,400],axis=0)
train_target = np.delete(dataset.target,[1,5,50,100,200,400])
test_data = dataset.data[[1,5,50,100,200,400]]
test_target = dataset.target[[1,5,50,100,200,400]]
#Train a model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(train_data,train_target)
#Testing the model
pred = lin_reg.predict(test_data)
plt.plot(test_target,color="red")
plt.plot(pred,color="green")
plt.show()
print(pred, test_target)
import numpy as np
np.mean((lin_reg.predict(data.data) - data.target) ** 2)
