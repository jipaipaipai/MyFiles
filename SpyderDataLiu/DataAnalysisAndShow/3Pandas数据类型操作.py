# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:40:52 2018

@author: haoyu
"""

import numpy as np
import pandas as pd

'''
重新索引
'''
dl={'城市':['北京','上海','广州','深圳','沈阳'],
    '环比':[101.5,101.2,101.3,102.0,100.1],
    '同比':[120.7,127.3,119.4,140.9,101.4],
    '定基':[121.4,127.8,120.0,145.5,101.6]}
d=pd.DataFrame(dl,index=['c1','c2','c3','c4','c5'])
d

d.reindex(index=['c5','c4','c3','c2','c1'])
d=d.reindex(columns=['城市','同比','环比','定基'])
d

newc=d.columns.insert(4,'新增')
newd=d.reindex(columns=newc,fill_value=200)
newd
'''
索引类型
'''
d.index
d.columns

nc=d.columns.delete(2)
ni=d.index.insert(5,'c0')
nd=d.reindex(index=ni,columns=nc,method='ffill')
nd

'''
删除指定索引对象
'''
a=pd.Series([9,8,7,6],index=['a','b','c','d'])
a
a.drop(['b','c'])

d
d.drop('c5')
d.drop('同比',axis=1)


'''
数据类型运算
'''
a=pd.DataFrame(np.arange(12).reshape(3,4))
a
b=pd.DataFrame(np.arange(20).reshape(4,5))
b

a+b
a*b

#方法型运算
b.add(a,fill_value=100)
a.mul(b,fill_value=0)

#不同类型数据运算
b=pd.DataFrame(np.arange(20).reshape(4,5))
b
c=pd.Series(np.arange(4))
c

c-10
b-c
b.sub(c,axis=0)

#比较运算
a=pd.DataFrame(np.arange(12).reshape(3,4))#同维运算
a
d=pd.DataFrame(np.arange(12,0,-1).reshape(3,4))
d
a>d
a==d

a=pd.DataFrame(np.arange(12).reshape(3,4))#不同维度运算
a
c=pd.Series(np.arange(4))
c
a>c
c>0


