import numpy as np
import pandas as pd

def get_samples(dataframe):
    random_df = dataframe.sample(frac=1)
    split_dataframe = np.array_split(random_df, 10)
    return split_dataframe

def changeToBin(dataframe, column_name):
    dataframe[column_name] = np.where(dataframe[column_name] != "BENIGN", 1, dataframe[column_name])
    dataframe[column_name] = np.where(dataframe[column_name] == "BENIGN", 0, dataframe[column_name])
    return dataframe

dataframe = pd.read_csv("./CSVs/Portmap_03-11.csv")
# print(dataframe)
dataframe = dataframe.loc[:, (dataframe != 0).any(axis=0)] # delete all 0 features
dataframe = dataframe.dropna(axis=1, how='all') # delete all NAN features
# print(dataframe)
# delete all irrelevant features
relevant_col = [col for col in dataframe.columns if 
    'Packet Length Mean' in col or 
    'Average Packet Size' in col or
    'Max Packet Length' in col or
    'Avg Fwd Segment Size' in col or
    'Fwd Packet Length Mean' in col or
    'Fwd Packet Length Max' in col or
    'Fwd Packet Length Min' in col or
    'Subflow Fwd Bytes' in col or
    'Total Length of Fwd Packets' in col or
    'Min Packet Length' in col or
    'Source Port' in col or
    'act_data_pkt_fwd' in col or
    'Flow Duration' in col or
    'Fwd Packets/s' in col or
    'Flow IAT Mean' in col or
    'Flow IAT Max' in col or
    'Fwd IAT Total' in col or
    'Fwd IAT Mean' in col or
    'Fwd IAT Max' in col or
    'Flow IAT Std' in col]

print(relevant_col)

# all values with type object
delete_objects = [col for col, dataframe in dataframe.dtypes.items() if dataframe == object]
for col in dataframe.columns: # delete all features of type object 
    if (col in delete_objects or col not in relevant_col) and col != " Label":
        dataframe.pop(col)

dataframe = dataframe.drop_duplicates().reset_index(drop=True)
# print(dataframe)
dataframe = changeToBin(dataframe, " Label")
# print(dataframe)
# samples = get_samples(dataframe)
# dataframe = samples[0]
# print(dataframe)
# dataframe.to_csv("test.csv")