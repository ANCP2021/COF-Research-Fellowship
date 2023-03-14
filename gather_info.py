import pandas as pd
import numpy as np
import glob

for infile in glob.glob("./CSVs/*.csv"):
    if infile == "./CSVs/UDPLag_01-12.csv":
        for chunk in pd.read_csv(infile, chunksize=100000): 
            print(chunk[' Label'].value_counts())