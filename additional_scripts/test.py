from __future__ import absolute_import, division, print_function, unicode_literals
import datetime
import tensorflow as tf
import os
import pandas as pd
import pathlib
import random
import shutil
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

data = pd.read_csv(r"/ProjectData/Data/train_multiclass_1.csv", sep=';',
                   thousands=",", low_memory=False)
del data["path_to_cell"]
# Вытаскиваем данные
X = data.iloc[:, 1:]
# Вытаскиваем лейблы
y = data["Label"]
X = np.array(X)

normalize = tf.keras.layers.experimental.preprocessing.Normalization()
model = tf.keras.Sequential([
    normalize,
    # tf.keras.layers.Dense(512,input_dim=702, activation='relu'),
    tf.keras.layers.Dense(702, activation='relu'),
    tf.keras.layers.Dense(702, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(11, activation='softmax')
])

model.compile(loss='mse',
              optimizer=tf.optimizers.Adam(), metrics='accuracy')
model.fit(X, y, epochs=1000, verbose=1)

test_loss, test_acc = model.evaluate(X, y, verbose=1)
print("Train = ", test_acc)

data = pd.read_csv(r"/ProjectData/Data/test_multiclass_1.csv", sep=';',
                   thousands=",", low_memory=False)
del data["path_to_cell"]
# Вытаскиваем данные
X = data.iloc[:, 1:]
# Вытаскиваем лейблы
y = data["Label"]
X = np.array(X)
test_loss, test_acc = model.evaluate(X, y, verbose=1)
print("Test = ", test_acc)
