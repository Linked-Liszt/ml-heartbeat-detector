
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
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
print(np.shape(full_data))
full_data += abnormal_data
print(np.shape(abnormal_data))
random.shuffle(full_data)

print(np.shape(full_data))

features = []
labels = []
for data in full_data:
    features.append(data[0])
    #labels.append(data[1])

print(np.shape(features))
print(np.shape(labels))




"""
model = tf.keras.models.Sequential()  # a basic feed-forward model
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))  
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))  
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam',  
              loss='binary_crossentropy',  
              metrics=['accuracy']) 
"""