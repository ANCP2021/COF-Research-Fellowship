import numpy as np
import pandas as pd

def get_samples(dataframe):
    random_df = dataframe.sample(frac=1)
    split_dataframe = np.array_split(random_df, 1000)
    return split_dataframe

def changeToBin(dataframe, column_name):
    dataframe[column_name] = np.where(dataframe[column_name] != "BENIGN", 1, dataframe[column_name])
    dataframe[column_name] = np.where(dataframe[column_name] == "BENIGN", 0, dataframe[column_name])
    return dataframe

dataframe = pd.read_csv("./CSVs/UDP_03-11.csv")

dataframe = dataframe.loc[:, (dataframe != 0).any(axis=0)] # delete all 0 features
dataframe = dataframe.dropna(axis=1, how='all') # delete all NAN features

# all values with type object
delete_objects = [col for col, dataframe in dataframe.dtypes.items() if dataframe == object]
for col in dataframe.columns: # delete all features of type object 
    if col in delete_objects and col != " Label":
        dataframe.pop(col) 


dataframe = dataframe.drop_duplicates().reset_index(drop=True)

dataframe = changeToBin(dataframe, " Label")
samples = get_samples(dataframe)
dataframe = samples[0]
print(dataframe)
# dataframe.to_csv("test.csv")