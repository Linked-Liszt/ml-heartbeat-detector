
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import pickle
import time
import random
import math

SPLIT = 0.8

with open('../compressedData/normal_f1_TF.pickle', 'rb') as normal_f:
    normal_data = pickle.load(normal_f)

with open('../compressedData/abnormal_f1_TF.pickle', 'rb') as abnormal_f:
    abnormal_data = pickle.load(abnormal_f)

full_data = normal_data
full_data += abnormal_data
random.shuffle(full_data)


features = []
labels = []
for data in full_data:
    newData = []
    for point in data[0]:
        newData.append([point])
    features.append(newData)
    labels.append(data[1])

print(np.shape(features))
print(np.shape(labels))

features = np.asarray(features)
labels = np.asarray(labels)

NAME = "cnn-3-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model = tf.keras.models.Sequential()

for i in range(2):
    model.add(Conv1D(filters=64, kernel_size=3, activation=tf.nn.relu, input_shape=(450, 1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation=tf.nn.relu))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(2))
model.add(Conv1D(filters=128, kernel_size=3, activation=tf.nn.relu, input_shape=(450, 1)))
model.add(Conv1D(filters=128, kernel_size=3, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(512, activation=tf.nn.relu))

"""
model.add(tf.keras.layers.Dense(450, input_shape=(450, ), activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
"""
model.add(Dense(1, activation=tf.nn.sigmoid))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

model.fit(features, labels, validation_split=0.3, epochs=15, callbacks=[tensorboard])