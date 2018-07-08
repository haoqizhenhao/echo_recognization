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


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", 0.0, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

FLAGS = tf.flags.FLAGS