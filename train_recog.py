# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 19:48:00 2018
模式识别算法：
深海采矿时，经常用超声探测待开采区的底质，本算法用于识别混响环境下的钴结壳底质、基岩（玄武岩、花岗岩等），泥沙底质
@author: haoqi
"""

import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Input, Flatten, Dense, MaxPooling1D, Dropout
from keras.layers.convolutional import Conv1D
from keras.utils.np_utils import to_categorical
from keras.models import Model
from echo_utils import load_dataset

# 导入数据
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
x_train = train_set_x_orig.reshape(-1,700,1)
x_test = test_set_x_orig.reshape(-1,700,1)
#x_train = train_set_x_orig
#x_test = test_set_x_orig
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255 
y_train = keras.utils.to_categorical(train_set_y_orig.squeeze(), 4)
y_test = keras.utils.to_categorical(test_set_y_orig.squeeze(), 4)

# build the model
x = Input(shape=(700,1))
y = Conv1D(16,kernel_size=7,strides=1,padding='valid',activation='relu',kernel_initializer='uniform')(x)
y = MaxPooling1D(pool_size=2)(y)

y = Conv1D(32,kernel_size=3,strides=1,padding='valid',activation='relu',kernel_initializer='uniform')(y)
y = MaxPooling1D(pool_size=2)(y)

y = Conv1D(64,kernel_size=3,strides=1,padding='valid',activation='relu',kernel_initializer='uniform')(y)
y = Conv1D(64,kernel_size=3,strides=1,padding='valid',activation='relu',kernel_initializer='uniform')(y)
y = MaxPooling1D(pool_size=2)(y)

y = Flatten()(y)
y = Dense(64,activation='relu')(y)
y = Dropout(0.5)(y)
y = Dense(32,activation='relu')(y)
y = Dropout(0.5)(y)
y = Dense(4,activation='softmax')(y)

model = Model(inputs=x, outputs=y, name='model')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

h = model.fit(x_train, y_train, epochs=200, batch_size=8, shuffle=True)
acc = h.history['acc']
m_acc = np.argmax(acc)
print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
loss, accuracy = model.evaluate(x_test, y_test, batch_size=20)
print('test loss: ',loss, 'test accuracy: ', accuracy)
## Parameters
## ==================================================
#
## Data loading params
#tf.flags.DEFINE_float("dev_sample_percentage", 0.0, "Percentage of the training data to use for validation")
#
## Model Hyperparameters
#tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
#tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
#tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
#tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
#
## Training parameters
#tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
#tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
#tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
#tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
#tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
#
#FLAGS = tf.flags.FLAGS