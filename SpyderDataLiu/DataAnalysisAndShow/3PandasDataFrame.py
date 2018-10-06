# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:18:54 2018

@author: haoyu
"""

import pandas as pd
import numpy as np
'''
DataFrame数据类型
'''
d=pd.DataFrame(np.arange(10).reshape(2,5))
d
#从ndarray对象字典创建
dt={'one':pd.Series([1,2,3],index=['a','b','c']),
    'two':pd.Series([9,8,7,6],index=['a','b','c','d'])}
d=pd.DataFrame(dt)
d
pd.DataFrame(dt,index=['b','c','d'],columns=['two','three'])
#从列表字典创建
dl={'one':[1,2,3,4],'two':[9,8,7,6]}
d=pd.DataFrame(dl,index=['a','b','c','d'])
d
#中文示例
dl={'城市':['北京','上海','广州','深圳','沈阳'],
    '环比':[101.5,101.2,101.3,102.0,100.1],
    '同比':[120.7,127.3,119.4,140.9,101.4],
    '定基':[121.4,127.8,120.0,145.5,101.6]}
d=pd.DataFrame(dl,index=['c1','c2','c3','c4','c5'])
d

d.index
d.columns
d.values

d['同比']
d.ix['c2']
d['同比']['c2']
