# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:25:09 2018

@author: haoyu
"""


import numpy as np
import pandas as pd
'''
数据排序
'''
#对索引进行排序
b=pd.DataFrame(np.arange(20).reshape(4,5),index=['c','a','d','b'])
b
b.sort_index()
b.sort_index(ascending=False)
c=b.sort_index(axis=1,ascending=False)
c

#对数据进行排序
b=pd.DataFrame(np.arange(20).reshape(4,5),index=['c','a','d','b'])
b
c=b.sort_values(2,ascending=False)
c
c=c.sort_values('a',axis=1,ascending=False)
c

a=pd.Series([9,8,7,6],index=['a','b','c','d'])
a
a.describe()
type(a.describe())
a.describe()['count']
a.describe()['max']

b=pd.DataFrame(np.arange(20).reshape(4,5),index=['c','a','d','b'])
b
b.describe()
type(b.describe())
b.describe().ix['max']
b.describe()[2]

'''
累计统计分析
'''
b=pd.DataFrame(np.arange(20).reshape(4,5),index=['c','a','d','b'])
b
b.cumsum()
b.cumprod()
b.cummin()
b.cummax()

b.rolling(2).sum()
b.rolling(3).sum()

'''
数据相关分析
'''
hprice=pd.Series([3.04,22.93,12.75,22.6,12.33],
                 index=['2008','2009','2010','2011','2012'])
m2=pd.Series([8.18,18.38,9.13,7.82,6.69],
             index=['2008','2009','2010','2011','2012'])
hprice.corr(m2)

import matplotlib.pyplot as plt
plt.plot(hprice,'',m2)

