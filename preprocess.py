import pandas as pd
import glob

appended_data = []
for infile in glob.glob("./CSVs/*.csv"):
    print(infile)
    dataframe = pd.read_csv(infile)
    print("beep 1")
    dataframe = dataframe.loc[:, (dataframe != 0).any(axis=0)] # delete all 0 features
    dataframe = dataframe.dropna(axis=1, how='all') # delete all NAN features
    # all values with type object
    delete_objects = [col for col, dataframe in dataframe.dtypes.items() if dataframe == object]
    # delete all irrelevant features
    irrelevant_count_columns = [col for col in dataframe.columns if 
                            'Flag Count' in col or 
                            'PSH' in col or 
                            '.' in col or 
                            "Bwd Segment" in col or
                            "Total Length of Bwd" in col or 
                            "Flow Bytes" in col or 
                            "Flow Packets" in col or
                            "Bwd IAT Total" in col or 
                            "Bwd IAT Std" in col or 
                            "Bwd IAT Min" in col or 
                            "Down" in col or 
                            "Subflow Bwd Bytes" in col or 
                            "Init_Win_bytes_backward" in col]
    print("beep 2")
    for col in dataframe.columns: # delete all features of type object 
        if col in delete_objects and col != " Label":
            dataframe.pop(col) 

        if "Idle" in col or "Active" in col or "Inbound" in col or "0" in col or "Bwd Packet Length" in col: # irrelevant features
            dataframe.pop(col) 
    
        if col in irrelevant_count_columns and col != " ACK Flag Count": # needed ACK Flag Count but not other _ Flag Count
            dataframe.pop(col)
    print("beep 3")
    dataframe = dataframe.drop_duplicates().reset_index(drop=True)
    print(dataframe)
    # store DataFrame in list
    appended_data.append(dataframe)
    print("appended")

# see pd.concat documentation for more info
appended_dataframe = pd.concat(appended_data)

# normalization
for column in appended_dataframe.columns:
    if column !=  " Label":
        appended_dataframe[column] = (appended_dataframe[column] - appended_dataframe[column].min()) / (appended_dataframe[column].max() - appended_dataframe[column].min())

# write DataFrame to a csv
print(appended_dataframe)

# appended_dataframe.to_csv("overall.csv", index=None, sep=',')