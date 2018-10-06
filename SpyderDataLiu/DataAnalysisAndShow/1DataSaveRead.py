# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:57:29 2018

@author: haoyu
"""

import numpy as np

#csv数据存取
a=np.arange(100).reshape(5,20)
np.savetxt('a.csv',a,fmt='%d',delimiter=',')

#csv数据读入
b=np.loadtxt('a.csv',delimiter=',')
b

#多维数据存取
a=np.arange(100).reshape(5,10,2)
a.tofile('b.dat',sep=',',format='%d')
c=np.fromfile('b.dat',dtype=np.int,sep=',')
c
c=np.fromfile('b.dat',dtype=np.int,sep=',').reshape(5,10,2)
c

#便捷文件存取
a=np.arange(100).reshape(5,10,2)
np.save('a.npy',a)
b=np.load('a.npy')
b

#numpy随机库
a=np.random.rand(3,4,5)
a
sn=np.random.randn(3,4,5)
sn
b=np.random.randint(100,200,(3,4))
b
np.random.seed(10)

#numpy统计运算
a=np.arange(15).reshape(3,5)
a
np.sum(a)
np.mean(a,axis=0)
np.mean(a,axis=1)
np.average(a,axis=0,weights=[10,5,1])#加权平均
np.std(a)
np.var(a)

b=np.arange(15,0,-1).reshape(3,5)
b
np.max(b)
np.argmax(b)
np.unravel_index(np.argmax(b),b.shape)
np.ptp(b)#极差
np.median(b)

#梯度
a=np.random.randint(0,20,(5))
a
np.gradient(a)
b=np.random.randint(0,50,(3,5))
b
np.gradient(b)

