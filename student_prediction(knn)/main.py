import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier



# reading data
data = pd.read_csv("student_prediction(knn)/StudentPerformanceFactors.csv")

# filter data
converter = {"Low":0,"High":2,"Medium":1,"No":0,"Yes":1,"Public":1,"Private":0,'Positive':1 ,'Negative':2 ,'Neutral':0,
             'High School':0, 'College':1 ,'Postgraduate':2}
ignored_column = ["Gender"]
for col in data.columns:
    print(col,data[col].unique())
    