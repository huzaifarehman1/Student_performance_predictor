import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score
import matplotlib.pyplot as plt
# parameters
PATH = "/home/huzaifa/code/AI/Machine_learning/student_prediction/StudentPerformanceFactors.csv"
ignore_column = ["Gender", "Distance_from_Home", "Teacher_Quality"]
report = True
percentage = 25
percentage = percentage/100


# data filtering
data = pd.read_csv(PATH)
data = data.drop(columns=ignore_column)
conversion = {"Low":0,"High":2,"Medium":1,"No":0,"Yes":1,"Public":1,"Private":0,
                  'Positive':1,'Negative':2,'Neutral':0,
                  'High School':0,'College':1,'Postgraduate':2, np.nan:1}
converter = lambda x: conversion[x]


for col in data.columns:
    if col not in ignore_column and isinstance(data[col][2], str):
        data[col] = data[col].apply(converter)
   
"""for i in data.columns:
    plt.scatter(data[i],data["Exam_Score"])
    plt.title(i)
    plt.xlabel(i)
    plt.ylabel("EXAM_SCORE")
    plt.show()
"""
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=percentage, random_state=42)

models = {
    
    "LINEAR_REGRESSION":LinearRegression(),
          }

if report:
    # make fit model
    
    
    best = { # evaluation:[name,score]
        "MAE":  (None, float("inf")),   # lower is better
        "MSE":  (None, float("inf")),   # lower is better
        "RMSE": (None, float("inf")),   # lower is better
        "R2":   (None, float("-inf")),  # higher is better
    }
    for name,model in models.items():
        print("\n")
        print(f"____{name}____")
        model = model.fit(x_train,y_train)
        
        #model.score() must be close to 1
        # test model
        y_pred = model.predict(x_test)
    
        MAE = mean_absolute_error(y_test, y_pred)
        MSE = mean_squared_error(y_test,y_pred)
        RMSE = root_mean_squared_error(y_test,y_pred)
        R2 = r2_score(y_test,y_pred)
        print("MAE",round(MAE,4))
        print("MSE",round(MSE,4))
        print("RMSA",round(RMSE,4))
        print("R^2",round(R2,4))
        
        metrics = {"MAE": MAE, "MSE": MSE, "RMSE": RMSE, "R2": R2}

        for metric_name, metric_value in metrics.items():
            if metric_name == "R2":
                # higher is better
                if metric_value > best[metric_name][1]:
                    best[metric_name] = (name, metric_value)
            else:
                # lower is better for MAE/MSE/RMSE
                if metric_value < best[metric_name][1]:
                    best[metric_name] = (name, metric_value)
                    
print("\n\n")
for parameter,card in best.items():
    print(f"Best {parameter} is {card[1]} for {card[0]}")            
                