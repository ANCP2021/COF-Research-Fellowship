import pandas as pd
import numpy as np
import glob
# BENIGN = BENIGN
# UDP-lag = UDP-lag
# NTP = DrDoS_NTP
# Syn = Syn
# SSDP = DrDoS_SSDP
# UDP = DrDoS_UDP, UDP
# NetBIOS = NetBIOS, DrDoS_NetBIOS
# MSSQL = MSSQL, DrDoS_MSSQL
# DNS = DrDoS_DNS
# SNMP = DrDoS_SNMP
# TFTP = TFTP
# LDAP = DrDoS_LDAP
# Portmap = Portmap
# WebDDoS = WebDDoS
webDDoS, benign, ntp, ssdp, snmp, dns, ldap, netBios, mssql, portmap, syn, tftp, udp, udp_lag = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
attack_type = ""
dataframe = pd.read_csv("./CSVs/Portmap_03-11.csv")
dataframe = dataframe.iloc[0:0]

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

for infile in glob.glob("./CSVs/*.csv"):
    for chunk in pd.read_csv(infile, chunksize=100000):

        for attack in chunk[" Label"]:
            if attack != 'BENIGN':
                attack_type = attack
                break

        chunk = chunk.loc[:, (chunk != 0).any(axis=0)] # delete all 0 features
        chunk = chunk.dropna(axis=1, how='all') # delete all NAN features
        # all values with type object
        delete_objects = [col for col, chunk in chunk.dtypes.items() if chunk == object]
        for col in chunk.columns: # delete all features of type object 
            if (col in delete_objects or col not in relevant_col) and col != " Label":
                chunk.pop(col)

        chunk = chunk.drop_duplicates().reset_index(drop=True)
        addition_to_dataframe = chunk.sample(n=10000, replace=False)
        for attack in addition_to_dataframe[" Label"]:
            
        exit()
        
