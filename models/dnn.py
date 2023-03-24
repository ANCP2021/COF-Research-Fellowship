import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


dataframe = pd.read_csv("./../preprocessing/data_bin.csv")
dataframe = dataframe[dataframe.columns.drop(list(dataframe.filter(regex='Unnamed')))]

x = dataframe.drop(' Label', axis=1)
y = dataframe[' Label']

ms = MinMaxScaler()
x = ms.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.3)

model = Sequential()
model.add(Dense(28 , input_shape=(X_train.shape[1],) , activation="relu" , name="Hidden_Layer_1"))
model.add(Dense(10 , activation="relu" , name="Hidden_Layer_2"))
model.add(Dense(1 , activation="sigmoid" , name="Output_Layer"))
opt = Adam(learning_rate=0.01)
model.compile( optimizer=opt, loss="binary_crossentropy", metrics=['accuracy'])
model.summary()

history_org = model.fit(
    X_train, 
    y_train, 
    batch_size=32, 
    epochs=100)