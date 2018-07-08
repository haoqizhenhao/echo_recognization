# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 23:00:56 2018
导入数据
@author: haoqi
"""

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import scipy.io

train_scale_of_all = 0.8
np.random.seed(1)
def data_input_orig():
    '''读取原数据
    点数：20000
    样本数：
    钴结壳：220，泥沙：200，基岩：104+101=205'''
    list_f = ['钴结壳/','泥/','玄武岩/']
    data = []
    num_gu = 0
    num_ni = 0
    num_yan = 0
    for rootdir in list_f:
        list_dir = os.listdir(rootdir)
        for ff in list_dir:
            for f in glob.glob(rootdir + ff + '/*.dat'):
                val = open(f,'r').read()
                val = val.split()
                data.append(val)
                if rootdir == list_f[0]:
                    num_gu += 1
                if rootdir == list_f[1]:
                    num_ni += 1
                if rootdir == list_f[2]:
                    num_yan += 1
    data = np.array(data).astype('float')
    return data, num_gu, num_ni, num_yan
##data, num_gu, num_ni, num_yan = data_input()

def data_input_echo():
    dir_data = 'data/'
    list_dir = os.listdir(dir_data)
    cc = np.array([])
    for i in list_dir:
        data = scipy.io.loadmat(dir_data+'/'+i)  # 读取mat文件
        data = data['ECHO']['Data']
        
        for j in np.arange(data.shape[1]):
            bb = data[0][j][0]
            cc = np.concatenate((cc,bb))
    data = cc.reshape(625,700)
#    data_gu = data[0:220,:]
#    data_ni = data[220:420,:]
#    data_yan1 = data[420:524,:]
#    data_yan2 = data[524::,:]
    # 构造标签
    label_gu = np.zeros([220])
    label_ni = np.ones([200])
    label_yan1 = 2*np.ones([104])
    label_yan2 = 3*np.ones([101])
    label = np.concatenate((label_gu, label_ni, label_yan1, label_yan2))
    # 打乱数据
    permutation = np.random.permutation(data.shape[0])
    shuffled_dataset = data[permutation, :]
    shuffled_labels = label[permutation]
    return shuffled_dataset, shuffled_labels
data, label = data_input_echo()



def h5_trans(data, label):
    train_file_name = 'train_data_echo.h5'
    f = h5py.File(train_file_name, "w") #用’w’模式打开文件
    f["train_set_x"] = data[0:500,:]
    f["train_set_y"] = label[0:500]
    test_file_name = 'test_data_echo.h5'
    f1 = h5py.File(test_file_name, "w") #用’w’模式打开文件
    f1["test_set_x"] = data[500::,:]
    f1["test_set_y"] = label[500::]
    f1["list_classes"] = [0,1,2,3]
    f.close()
    f1.close()
#    f = h5py.File(train_file_name, "w")
#    f["train_set_y"] = label[0:500,:]
#    return f
h5_trans(data, label)


    #dset = f.create_dataset("mydataset", (data.shape[0],), dtype='i') #create_dataset用于创建给定形状和数据类型的空datase