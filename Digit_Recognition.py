# Importing Dependencies

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D,Flatten, Dropout

# Callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Loading the Dataset 
(X_train,y_train), (X_test,y_test) = mnist.load_data()

X_train.shape, y_train.shape, X_test.shape, y_test.shape

def plot_input_image(i):
    plt.imshow(X_train[i],cmap='binary',interpolation='nearest')
    plt.title(y_train[i])
    plt.axis('off')
    plt.show()

# Pre-processing

# Normalizing the image to [0,1] range

X_train = X_train.astype(np.float32)/255
X_test = X_test.astype(np.float32)/255

# Reshaping or Expanding the dimensions of image from
# (28,28) to (28,28,1)

X_train = np.expand_dims(X_train,-1)
X_test = np.expand_dims(X_test,-1)

# One-hot encoding

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Building the Model

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())

model.add(Dropout(0.25))
# 25 % value is dropped ; can also try with value 0.5 for 50 % to be dropped

model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()

model.compile(optimizer='adam',loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])

# EarlyStopping
es = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=4, verbose=1)

# Model Checkpoint
mc = ModelCheckpoint("./model.h5", monitor='val_accuracy', verbose=1, save_best_only=True)

cb = [es,mc]

# Model Training 
model.fit(X_train, y_train, epochs=10, validation_split=0.3, callbacks=cb)

loss,accuracy = model.evaluate(X_test,y_test)
print(f"The Loss of the model is {loss}")
print(f"The Accuracy of the model is {accuracy*100}")