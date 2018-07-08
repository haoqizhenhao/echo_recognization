# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 19:48:00 2018
模式识别算法：
深海采矿时，经常用超声探测待开采区的底质，本算法用于识别混响环境下的钴结壳底质、基岩（玄武岩、花岗岩等），泥沙底质
@author: haoqi
"""

import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

data = scipy.io.loadmat('ECHO.mat')  # 读取mat文件

data = data['ECHO']['Data']
cc = np.array([])
for i in np.arange(data.shape[1]):
    bb = data[0][i][0]
    cc = np.concatenate((cc,bb))
    
