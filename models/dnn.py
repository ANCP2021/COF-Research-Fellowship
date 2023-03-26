import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataframe = pd.read_csv("./../preprocessing/data_bin.csv")
dataframe = dataframe[dataframe.columns.drop(list(dataframe.filter(regex='Unnamed')))]

# load data
x = dataframe.drop(' Label', axis=1)
y = dataframe[' Label']

# normalize
ms = MinMaxScaler()
x = ms.fit_transform(x)

# split training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3)

# define keras model
model = Sequential()
model.add(Dense(21, input_shape=(X_train.shape[1], ), activation="relu", name="Input_Layer"))
model.add(Dense(15, activation="relu", name="Hidden_Layer"))
model.add(Dense(1, activation="sigmoid", name="Output_Layer"))
# optimize
optimize = Adam(learning_rate=0.01)
# compile model
model.compile(optimizer=optimize, loss="binary_crossentropy", metrics=['accuracy'])
model.summary()

history_org = model.fit(
    X_train, 
    y_train, 
    batch_size=32, 
    epochs=200, 
    verbose=2, 
    callbacks=None, 
    validation_data=(X_test, y_test), 
    shuffle=True, 
    class_weight=None, 
    sample_weight=None, 
    initial_epoch=0)

# evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print('Accuracy of Deep neural Network : %.2f' % (accuracy*100))

loss = history_org.history['loss']
val_loss = history_org.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g', label = 'Training Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.title('Loss v/s No. of epochs')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

loss = history_org.history['accuracy']
val_loss = history_org.history['val_accuracy']
plt.plot(epochs, loss, 'g', label = 'Training accuracy')
plt.plot(epochs, val_loss, 'r', label = 'Validation accuracy')
plt.title('Accuracy Scores v/s Number of Epochs')
plt.xlabel('No. of Epochs')
plt.ylabel('Accuracy Score')
plt.legend()
plt.show()