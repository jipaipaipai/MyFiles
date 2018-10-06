# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 14:08:08 2018

@author: haoyu
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X_data = np.linspace(-0.5, 0.5, 200).reshape(-1, 1)
noise = np.random.normal(0, 0.02, X_data.shape)
y_data = np.square(X_data) + noise

#定义两个placeholder
X = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

#定义神经网络中间层
Weight_L1 = tf.Variable(tf.random_normal([1, 10]))#输入层是1个神经元（1个值），中间层是10个神经元
biases_L1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_L1 = tf.matmul(X, Weight_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)#获得中间层L1的输出

#定义神经网络输出层
Weight_L2 = tf.Variable(tf.random_normal([10, 1]))#中间层为10，输出层为1
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weight_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

#二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
#使用梯度
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step, feed_dict={X:X_data, y:y_data})
    #获得预测值
    prediction_value = sess.run(prediction, feed_dict={X:X_data})
    #画图
    plt.figure()
    plt.scatter(X_data, y_data)
    plt.plot(X_data, prediction_value, 'r-', lw=5)
    plt.show()































