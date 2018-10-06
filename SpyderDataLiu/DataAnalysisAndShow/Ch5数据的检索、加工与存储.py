# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 20:07:55 2018

@author: haoyu
"""


import numpy as np
import pandas as pd

'''
利用numpy和pandas对CSV文件进行写操作
'''

np.random.seed(42)
a=np.random.randn(3,4)
a[2][2]=np.nan
print(a)
a

np.savetxt('np.csv',a,fmt='%.2f',delimiter=',',header='#1,#2,#3,#4')

df=pd.DataFrame(a)
df
df.to_csv('pd.csv',float_format='%.2f',na_rep='NAN!')

'''
numpy.npy与pandas DataFrame
'''
from tempfile import NamedTemporaryFile
from os.path import getsize

np.random.seed(42)
a=np.random.randn(365,4)

tmpf=NamedTemporaryFile()
np.savetxt(tmpf,a,delimiter=',')
getsize(tmpf.name)

tmpf=NamedTemporaryFile()
np.save(tmpf,a)
tmpf.seek(0)
loaded=np.load(tmpf)
print(loaded.shape)
getsize(tmpf.name)

df=pd.DataFrame(a)
df.to_pickle(tmpf.name)
getsize(tmpf.name)
pd.read_pickle(tmpf.name)

'''
使用pandas读写excel文件
'''
from tenpfile import NamedTemporaryFile

np.random.seed(42)
a=np.random.randn(365,4)

tmpf=NamedTemporaryFile(suffix='xlsx')
df=pd.DataFrame(a)
print(tmpf.name)
df.to_excel(tmpf.name,sheet_name='Random Data')
print('Means\n',pd.read_excel(tmpf.name,'Random Data').mean())




