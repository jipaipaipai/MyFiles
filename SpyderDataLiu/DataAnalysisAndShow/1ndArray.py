# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:40:12 2018

@author: haoyu
"""
#创建数组
import numpy as np
a=np.array([[0,1,2,3,4],
           [9,8,7,6,5]])
a
a.ndim
a.shape
a.size
a.dtype
a.itemsize

np.arange(10)
np.ones((2,3))
np.zeros((3,6),dtype=np.int32)
np.eye(5)

a=np.linspace(1,10,4)
a
b=np.linspace(1,10,4,endpoint=False)
b
c=np.concatenate((a,b))
c

a=np.ones((2,3,4),dtype=np.int32)
a.reshape((3,8))
a
a.resize((3,8))
a

a=np.ones((2,3,4),dtype=np.int)
a
b=a.astype(np.float)
b

a=np.full((2,3,4),25,dtype=np.int32)
a
a.tolist()


#数组的索引
a=np.array([9,8,7,6,5])
a[2]
a[1:4:2]

a=np.arange(24).reshape((2,3,4))
a
a[1,2,3]
a[-1,-2,-3]
a[:,1,-3]
a[:,1:3,:]
a[:,:,::2]

#数组与标量计算
a=np.arange(24).reshape((2,3,4))
a
a.mean()
a=a/a.mean()
a
np.sign(a)

a=np.arange(24).reshape((2,3,4))
np.square(a)
a=np.sqrt(a)
a
np.modf(a)#分开数组中的小数和证书部分

a=np.arange(24).reshape((2,3,4))
b=np.sqrt(a)
a
b
np.maximum(a,b)
a>b

#合并操作
x=np.array([1,2,3])
y=np.array([3,2,1])
np.concatenate([x,y])

A=np.array([[1,2,3],
            [4,5,6]])
np.concatenate([A,A])
np.concatenate([A,A],axis=1)

np.concatenate([A,x.reshape(1,3)])
np.concatenate([A,np.array([[1,2]]).reshape(2,1)],axis=1)

np.vstack([A,x])
np.hstack([A,np.array([[1,2]]).reshape(2,1)])

#分割操作
x=np.arange(10)
x1, x2, x3=np.split(x, [ 3, 7])
x1
x2
x3
x1,x2=np.split(x,[5])
x1
x2

A=np.arange(16).reshape(4,4)
A1,A2=np.split(A,[2])
A1,A2=np.split(A,[2],axis=1)
left,right=np.hsplit(A,[2])



##numpy.array中的运算
a1=[i for i in range(10)]
a2=np.arange(10)
a1*2
a2*2

##unicersal function
a=np.arange(16).reshape(4,4)
a+1
a**2
a//2

np.log(a)
pow(3,a)
a*a

##矩阵乘法
a.dot(a)

###矩阵的逆
inva=np.linalg.inv(a)
inva.dot(a)



##聚合操作
L=np.random.random(100)
sum(L)
np.sum(L)
L.sum()

np.sum(a)
np.sum(a,axis=0)

np.prod(a+1)
np.mean(a)

##索引arg
d=np.random.normal(0,1,1000000)
np.min(d)
np.argmin(d)
d[692659]

#排序和使用索引
s=np.arange(16)
np.random.shuffle(s)
s
np.sort(s)
np.argsort(s)
s
s.sort()
s

ss=np.arange(16).reshape(4,4)
np.sort(ss)


##fancy indexing
ind=[3,5,8]
s[ind]

row=np.array([0,1,2])
col=np.array([1,2,3])
ss[row,col]

col=[True,True,False,True]
ss[1:3,col]
