import pandas as pd
import numpy as np

# check if duplicate value exists in dataframe
def have_duplicate(dataframe:pd.DataFrame):
    """check if dataframe have duplicate and return non single counts as well

    Args:
        dataframe (_type_): dataframe of dataset
    Returns:
        (bool,dict) : bool representing if it has duplicate and  dict having counter of non-single entries
    """
    counter = {}
    for i,row in dataframe.iterrows():
        s = ""
        for j in dataframe.columns:
            s += str(row[j])
        if s in counter:
            counter[s] += 1
        else:
            counter[s] = 1        
    new = {}        
    ans = False
    for k,v in counter.items():
        if v>1:
           new[k] = v
           ans = True
    del counter                
    return ans,new

dt = pd.read_csv("student_prediction/StudentPerformanceFactors.csv")        
print(have_duplicate(dt))
#duplicates_exist = dt.duplicated().any()
#print(duplicates_exist)
