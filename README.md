# Ex.No.5-Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas library to read csv or excel file.

2.Import LabelEncoder using sklearn.preprocessing library.

3.Transform the data's using LabelEncoder.

4.Import decision tree classifier from sklearn.tree library to predict the values.

5.Find accuracy.

6.Predict the values.

7.End of the program.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sriram G
RegisterNumber:  212222230149
*/
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("Placement_Data.csv")
dataset
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))
def gradient_descent (theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta
theta =  gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X): 
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```

## Output:

### Dataset:
![image](https://github.com/Sriram8452/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118708032/16c9b5d5-e699-49f4-a8ef-9362e0ee16f3)

### Dataset.dtypes:
![image](https://github.com/Sriram8452/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118708032/3c62baba-7676-422b-bd42-04f637e3808b)

### Labeled_dataset:
![image](https://github.com/Sriram8452/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118708032/eb7e98f5-45d6-4c2d-b8b8-0367bf4850ef)

### Dependent variable Y:

![image](https://github.com/Sriram8452/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118708032/18f6cb58-669e-4ec7-a94f-a28fdd50abe4)

### Accuracy:
![image](https://github.com/Sriram8452/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118708032/f596e0a6-21fc-476a-b11c-5042b188f66b)

### y_pred:
![image](https://github.com/Sriram8452/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118708032/f20dff88-d594-4e00-9107-c0cb5d4dabdf)

## Y:

![image](https://github.com/Sriram8452/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118708032/09644f7a-1c28-445d-a9ab-bd520bd01aa5)

### y_pred:
![image](https://github.com/Sriram8452/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118708032/65196d4d-d220-4dee-8fe4-798bbb9ebd7c)
![image](https://github.com/Sriram8452/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118708032/41e9cc21-fd0f-4d47-8e50-37ed8fb6a27f)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

