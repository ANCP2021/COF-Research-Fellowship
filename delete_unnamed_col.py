import pandas as pd
import numpy as np

dataframe = pd.read_csv("./data.csv")
dataframe = dataframe[dataframe.columns.drop(list(dataframe.filter(regex='Unnamed')))]
print(dataframe)
dataframe.to_csv("./data.csv")