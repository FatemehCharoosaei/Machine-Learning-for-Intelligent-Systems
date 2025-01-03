# -*- coding: utf-8 -*-


import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn import datasets
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback

import time


NAME = "NN {}".format(int(time.time()))
tensor_board = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=1)

class LogHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        self.weights1 = []
        self.weights2 = []

    def on_batch_end(self, batch, logs={}):
        self.weights1.append(self.model.layers[1].get_weights())
        self.weights2.append(self.model.layers[2].get_weights())
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))

        

path = ''

train_file = 'fashion-mnist_train.csv'
train_df = pd.read_csv(os.path.join('', train_file))
train_data = train_df.loc[:, 'pixel1':'pixel784'].values
train_label = train_df.loc[:, 'label'].values
train_data_norm = train_data/255.0


test_file = 'fashion-mnist_test.csv'
test_df = pd.read_csv(os.path.join(path, test_file))
test_data = test_df.loc[:, 'pixel1':'pixel784'].values
test_label = test_df.loc[:, 'label'].values
test_data_norm = test_data/255.0


img1 = train_data[3, :]
img1 = img1.reshape(-1, 1).reshape(28, 28)
plt.imshow(img1, cmap = 'gray')
plt.show

pca = PCA(n_components = 50)
pca.fit(train_data)

train_features = pca.transform(train_data)
test_features = pca.transform(test_data)


model = keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(40, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
history = LogHistory()
model.fit(train_features, train_label, batch_size=200, epochs = 5, validation_split=0.1, callbacks=[history, tensor_board] )

val_loss, val_accu = model.evaluate(test_features, test_label)
print(val_loss, val_accu)



predictions = model.predict(test_features)
output_class = np.argmax(predictions, axis=1)
err = mean_absolute_error(output_class, test_label)
print(err)

print(test_label[0])

#img1 = test_data[3, :]
#img1 = img1.reshape(-1, 1).reshape(28, 28)
#plt.imshow(img1, cmap = 'gray')
#plt.show

weights1 = history.weights1
w1 = []
w2 = []
for w, b in weights1:
    w1.append(w[0, 0])  
    w2.append(w[1, 0])

plt.plot(w1)
plt.plot(w2)
plt.show()


plt.plot(history.acc)
plt.show()

plt.plot(history.losses)
plt.show()

