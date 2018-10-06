# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 14:44:25 2018

@author: haoyu
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#载入数据集
mnist = input_data.read_data_sets(r'D:\Anaconda3\SpyderDataLiu\PythonDeepLearning\MNIST_data',
                                  one_hot=True)
#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examplse // batch_size

#定义两个placeholder
X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
#创建一个简单的神经网络
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(X, W) + b)
#二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
init = tf.global_variables_initializer()
#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction))#返回一维张量中最大的只所在的位置
#求准确路
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.next_batch(batch_size)
            sess.run(train_step, feed_dict={X:batch_xs, y:batch_ys})
        acc = sess.run(accuracy, feed_dict={X:mnist.test.images, y:mnist.test.labels})
        print('Iter'+str(epoch)+',Testing Accuracy'+str(acc))



































