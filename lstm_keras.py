# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 23:24:46 2018
LSTM
@author: haoqi
"""
import keras
import numpy as np
np.random.seed(2017)  #为了复现
#from __future__ import print_function
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense
from keras.optimizers import Adam
from echo_utils import load_dataset

#(X_train, y_train), (X_test, y_test) = mnist.load_data()

#参数
#学习率
learning_rate = 0.001 
#迭代次数
epochs = 500
#每块训练样本数
batch_size = 128
#输入
n_input = 28
#步长
n_step = 28
#LSTM Cell
n_hidden = 128
#类别
n_classes = 10

#x标准化到0-1  y使用one-hot  输入 nxm的矩阵 每行m维切成n个输入
#X_train = X_train.reshape(-1, n_step, n_input)/255.
#X_test = X_test.reshape(-1, n_step, n_input)/255.
#
#y_train = np_utils.to_categorical(y_train, num_classes=10)
#y_test = np_utils.to_categorical(y_test, num_classes=10)

# 导入数据
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
X_train = train_set_x_orig.reshape(-1,10,70)
X_test = test_set_x_orig.reshape(-1,10,70)
#x_train = train_set_x_orig
#x_test = test_set_x_orig
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#x_train /= 255
#x_test /= 255 
y_train = keras.utils.to_categorical(train_set_y_orig.squeeze(), 4)
y_test = keras.utils.to_categorical(test_set_y_orig.squeeze(), 4)

model = Sequential()
model.add(LSTM(n_hidden,
               batch_input_shape=(None, 10, 70),
               unroll=True))

model.add(Dense(4))
model.add(Activation('softmax'))

adam = Adam(lr=learning_rate)
#显示模型细节
model.summary()
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1, #0不显示 1显示
          validation_data=(X_test, y_test))

scores = model.evaluate(X_test, y_test, verbose=0)
print('LSTM test score:', scores[0]) #loss
print('LSTM test accuracy:', scores[1])