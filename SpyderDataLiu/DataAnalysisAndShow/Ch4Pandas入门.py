# -*- coding: utf-8 -*-
"""
Created on Thu May 31 19:00:47 2018

@author: haoyu
"""

import pandas as pd
import numpy as np
from pandas.io.parsers import read_csv
df=read_csv()

a=pd.DataFrame(np.arange(12).reshape(4,3))
b=pd.DataFrame(np.arange(3,15).reshape(4,3))

'''
利用Pandas的DataFrame实现数据整合
'''

from numpy.random import seed
from numpy.random import rand
from numpy.random import random_integers

seed(42)
df=pd.DataFrame({'Weather':['cold','hot','cold','hot','cold','hot','cold'],
                 'Food':['soup','soup','icecream','chocolate','icecream','icecream','soup'],
                 'Price':10*rand(7),'Number':random_integers(1,9,size=(7,))})
df

#单列分类
weather_group=df.groupby('Weather')
i=0
for name,group in weather_group:
    i=i+1
    print('Group',i,name)
    print(group)
name
weather_group.first()
weather_group.last()
weather_group.mean()

#多列进行分类
wf_group=df.groupby(['Weather','Food'])
wf_group.groups
wf_group.agg([np.mean,np.median])

'''
DataFrame的串联与附加操作
'''
df[:3]
pd.concat([df[:3],df[3:]])
df[:3].append(df[5:])
df[['Food','Weather']][0:1]

'''
处理日期数据
'''
pd.date_range('1/1/1900',periods=42,freq='D')

pd.to_datetime(['19991111','19651211'],format='%Y%m%d')

'''
数据透视
'''
pd.pivot_table(df,columns=['Food'],aggfunc=np.sum)



