import pandas as pd
import numpy as np

def changeToBin(dataframe, column_name):
    dataframe[column_name] = np.where(dataframe[column_name] != "BENIGN", 1, dataframe[column_name])
    dataframe[column_name] = np.where(dataframe[column_name] == "BENIGN", 0, dataframe[column_name])
    return dataframe

dataframe = pd.read_csv("./data.csv")
dataframe = dataframe[dataframe.columns.drop(list(dataframe.filter(regex='Unnamed')))]
dataframe = changeToBin(dataframe, ' Label')
print(dataframe)
dataframe.to_csv("./data_bin.csv")