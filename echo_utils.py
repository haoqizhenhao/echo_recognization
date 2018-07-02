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

def data_input():
    list_f = ['钴结壳/','泥/','玄武岩/']
    data = []
    for rootdir in list_f:
        list_dir = os.listdir(rootdir)
        for ff in list_dir:
            for f in glob.glob(rootdir + ff + '/*.dat'):
                val = open(f,'r').read()
                val = val.split()
                data.append(val)
    data = np.array(data).astype('float')
    return data