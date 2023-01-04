import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD

model = Sequential([Dense(5, activation='sigmoid', input_shape=(22,)), Dense(1, activation='sigmoid')])
opt = SGD(lr=0.07, momentum = 0.9)
model.compile(optimizer=opt, loss='binary_crossentropy')
model.fit(data_train, target_train, epochs=500, batch_size=2)
predicted1 = model.predict_classes(data_test)
print(accuracy_score(target_test, predicted1))
print(average_precision_score(target_test, predicted1))
