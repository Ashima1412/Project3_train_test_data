# Project3_train_test_data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

df1 = pd.read_csv('train.csv')

print(df1.shape)
df1.head()

df2 = pd.read_csv('test.csv')
print(df2.shape)

df2.head()

#label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1['species']=le.fit_transform(df1['species'])
df1.head()

X = df1.iloc[:,2:]
y = df1['species']
X.head()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# Decision Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
print('Accuracy is: ',metrics.accuracy_score(y_pred,y_test))

# Random Forest

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
print('Accuracy is: ',metrics.accuracy_score(y_pred,y_test))

# SVM

from sklearn.svm import LinearSVC
sc = LinearSVC()
sc.fit(X_train,y_train)
y_pred = sc.predict(X_test)
print('Accuracy is: ',metrics.accuracy_score(y_pred,y_test))

# Naive Bayes

from sklearn.naive_bayes import GaussianNB
gb = GaussianNB()
gb.fit(X_train,y_train)
y_pred = gb.predict(X_test)
print('Accuracy is: ',metrics.accuracy_score(y_pred,y_test))

# Best Classifier - svm
predict_test = sc.predict(df2.iloc[:,1:])
predict_test

# labelencoder.inverse_transform(predict_test) : Inverse the encoding
df2["species"] = le.inverse_transform(predict_test)
df2.head()
