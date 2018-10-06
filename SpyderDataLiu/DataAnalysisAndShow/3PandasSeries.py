# -*- coding: utf-8 -*-
"""
Created on Mon May 28 14:36:45 2018

@author: haoyu
"""

import pandas as pd

d=pd.Series(range(20))
d
d.cumsum()

'''
Series类型
'''

#列表创建
a=pd.Series([9,8,7,6])
a
b=pd.Series([9,8,7,6],index=['a','b','c','d'])
b

#从标量值进行创建
s=pd.Series(25,index=['a','b','c'])
s

#从字典创建
d=pd.Series({'a':9,'b':8,'c':7})
d
e=pd.Series({'a':9,'b':8,'c':7},index=['c','a','b','d'])
e#自定顺序

#从ndarray创建
import numpy as np
n=pd.Series(np.arange(5))
n
m=pd.Series(np.arange(5),index=np.arange(9,4,-1))
m

'''
基本操作
'''
b=pd.Series([9,8,7,6],index=['a','b','c','d'])
b
b.index
b.values
b['b']
b[1]
b[['c','d',0]]
b[['c','d','a']]
#类似array
b[3]
b[:3]
b[b>b.median()]
np.exp(b)
#类似python
b['b']
'c' in b
0 in b
b.get('f',100)
#对齐操作
a=pd.Series([1,2,3],['c','d','e'])
b=pd.Series([9,8,7,6],index=['a','b','c','d'])
a+b

#name属性
b=pd.Series([9,8,7,6],index=['a','b','c','d'])
b.name
b.name='Series对象'
b.index.name='索引列'
b
#Series类型修改
b=pd.Series([9,8,7,6],index=['a','b','c','d'])
b['a']=15
b.name='Series'
b
b.name='New Series'
b['b','c']=20
b



