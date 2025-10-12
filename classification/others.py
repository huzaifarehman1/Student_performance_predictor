# prerequisite
import numpy as np
import pandas as pd
# skleanr 
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
#balancer
from imblearn.over_sampling import RandomOverSampler


percentage = 25
percentage = percentage/100
ignored_column = ["Gender", "Distance_from_Home", "Teacher_Quality", "Tutoring_Sessions"]
want_report_logreg = False
want_report_svc = True
PATH = "/home/huzaifa/code/AI/Machine_learning/StudentPerformanceFactors.csv"

# loading data
data = pd.read_csv(PATH)
data = data.drop(columns=ignored_column)

# data filtering
def grader(n, Forward=True):
        x = [90,80,70,60,50,40,0]
        y = [0,1,2,3,4,5,6]
        if Forward:
            for i in range(len(x)):
                if n >= x[i]:
                    return y[i]
        else:
            for i in range(len(y)):
                if n == y[i]:
                    return (x[i]+10,x[i])

conversion = {"Low":0,"High":2,"Medium":1,"No":0,"Yes":1,"Public":1,"Private":0,
                  'Positive':1,'Negative':2,'Neutral':0,
                  'High School':0,'College':1,'Postgraduate':2, np.nan:1}
converter = lambda x: conversion[x]

data["Exam_Score"] = data["Exam_Score"].apply(grader)
for col in data.columns:
    if col not in ignored_column and isinstance(data[col][2], str):
        data[col] = data[col].apply(converter)

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=percentage, random_state=42)
ros = RandomOverSampler(random_state=42)
x_train_resampled, y_train_resampled = ros.fit_resample(x_train, y_train)

# report
if want_report_logreg:
    # model  logistic regression
    LOG_REG_model = LogisticRegression(class_weight="balanced")
    LOG_REG_model = LOG_REG_model.fit(x_train_resampled,y_train_resampled)
    y_pred = LOG_REG_model.predict(x_test)
    print(classification_report(y_test,y_pred))


# report
if want_report_svc:
    sv_model = SVC()
    sv_model = sv_model.fit(x_train_resampled,y_train_resampled)
    y_pred = sv_model.predict(x_test)
    print(classification_report(y_test,y_pred))    

