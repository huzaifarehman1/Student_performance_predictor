import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# reading data
data = pd.read_csv("student_prediction(knn)/StudentPerformanceFactors.csv")
percent = 25 #setup by user (for testing data)
neighbours = 6
# filter data
conversion = {"Low":0,"High":2,"Medium":1,"No":0,"Yes":1,"Public":1,"Private":0,'Positive':1 ,'Negative':2 ,'Neutral':0,
             'High School':0, 'College':1 ,'Postgraduate':2,np.nan:1}
converter = lambda x: conversion[x]

ignored_column = ["Gender","Distance_from_Home","Teacher_Quality"]

# Ignore some columns
data = data.drop(columns=ignored_column)

# Keep only rows with NO NaN values
#data = data.dropna()

def grader(n, Forward=True):
    x = [90,80,70,60,50,40,0]
    y = [0, 1, 2, 3, 4, 5, 6]
    if Forward:
        for i in range(len(x)):
            if n >= x[i]:
                return y[i]
    else:
        for i in range(len(y)):
            if n == y[i]:
                return (x[i]+10,x[i])
                 
           

            

data["Exam_Score"] = data["Exam_Score"].apply(grader)
for i in data.columns:
    if i not in ignored_column and isinstance(data[i][2],str):
        data[i] = data[i].apply(converter)



# create synthetic data for catagories





















# model part


size = percent / 100
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# Train-test split (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=size, random_state=42)

model = KNeighborsClassifier(n_neighbors=neighbours)

model.fit(x_train,y_train)

predictions = model.predict(x_test)

print(classification_report(y_test,predictions))