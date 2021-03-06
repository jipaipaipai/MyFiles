# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 20:17:13 2018

@author: haoyu
"""

'''创建图，启动图'''
import tensorflow as tf
#创建常量op
m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],[3]])
#创建一个矩阵乘法op，把m1和m2传入
product = tf.matmul(m1, m2)
print(product)#并不会输出结果

#定义一个会话，启动默认图
sess = tf.Session()
#调用sess的run方法来执行矩阵乘法op
#run(product)触发了图中3个op
result = sess.run(product)
print(result)
sess.close()#要关闭会话

#或者另一种方式创建会话
with tf.Session() as sess:
    #调用sess的ru方法来执行矩阵乘法op
    #run(product)触发了图中3个op
    result = sess.run(product)
    print(result)


'''变量'''
import tensorflow as tf
x = tf.Variable([1,2])
a = tf.constant([3,3])
#增加一个减法op
sub = tf.subtract(x,a)
#增加一个加法op
add = tf.add(x,sub)
#初始化全局变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))

##在tensorflow里循环
#创建一个变量初始化为0
state = tf.Variable(0,name='counter')
#创建一个op，作用是使state加1
new_value = tf.add(state, 1)
#赋值op,不能使用等于号直接赋值
update = tf.assign(state, new_value)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))


'''Fetch and Feed'''
import tensorflow as tf
#Fetch：同时run多个op
input1 = tf.constant(3.0)
input2 = tf.constant(4.0)
input3 = tf.constant(5.0)

add = tf.add(input2, input3)
mul = tf.multiply(input1, add)

with tf.Session() as sess:
    result = sess.run([mul, add])
    print(result)

#Feed
#创建占位符
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    #feed的数据以字典的形式传入
    print(sess.run(output, feed_dict={input1:[7.0], input2:[2.0]}))


'''tensorflow简单示例'''
import tensorflow as tf
import numpy as np

x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2

#构造一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b

#二次代价函数
loss = tf.reduce_mean(tf.square(y_data - y))#MSE
#定义一个梯度下降法来进行训练优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
#最小化代价函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 ==0:
            print(step, sess.run([k, b]))















































