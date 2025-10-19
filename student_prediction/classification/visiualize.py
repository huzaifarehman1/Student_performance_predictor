# prerequisite
import numpy as np
import pandas as pd
# matplotlib
import matplotlib.pyplot as plt



PATH = "/home/huzaifa/code/AI/Machine_learning/StudentPerformanceFactors.csv"

ignore_column = []

# load data

data = pd.read_csv(PATH)



# apply plot
conversion = {"Low":0,"High":2,"Medium":1,"No":0,"Yes":1,"Public":1,"Private":0,
                  'Positive':1,'Negative':2,'Neutral':0,
                  'High School':0,'College':1,'Postgraduate':2, np.nan:4}

converter = lambda x:conversion[x]
for col in data.columns:        
    if col not in ignore_column:
        if isinstance(data[col][2], str):
            data[col] = data[col].apply(converter)
            plt.title(f"{col} -> EXAM_SCORE")
            values = data[col].unique()
            occurence = [len(data[data[col] == val]) for val in values]
            plt.bar(values, occurence)
            plt.show()
            
        
        else:
            plt.title(col)
            plt.scatter(data[col].to_numpy().ravel(),data["Exam_Score"].to_numpy().ravel())
            plt.show()