# -*- coding: utf-8 -*-
"""
Created on Sun May 27 18:49:48 2018

@author: haoyu
"""
#围棋棋谱正方形个数

import numpy as np
import time
t1=time.perf_counter()
a=np.arange(361).reshape((19,19))
count=0
for i in range(19):
    for j in range(19):        
        b=np.where(a==a[i][j])
        if 18-b[0][0]<18-b[1][0]:
            count=count+(18-b[0][0])
        if 18-b[0][0]>18-b[1][0]:
            count=count+(18-b[1][0])
        if 18-b[0][0]==18-b[1][0]:
            count=count+(18-b[0][0])

t2=time.perf_counter()
t=t2-t1
print(count,t,sep='\n')

#日期计算

d=input()
while d !='':   
    a=eval(d[:4])
    days=0
    if (a%100==0 and a%400==0)or(a%100!=0 and a%4==0):
        monthday=[31,29,31,30,31,30,31,31,30,31,30,31]
        for i in range(int(eval(d[5:7]))-1):
            days=days+monthday[i]
        days=days+int(eval(d[-2:]))
    else:
        monthday=[31,28,31,30,31,30,31,31,30,31,30,31]
        for i in range(int(eval(d[5:7]))-1):
            days=days+monthday[i]
        days=days+int(eval(d[-2:]))
    print(days)
    d=input()

##面向对象
class Solution:
    def convert(self,s,numRows):
        '''
        :type s:str
        :type numRows:int
        :rtype:str
        '''
        n=numRows
        res_list=[]
        l=len(s)
        if n==1:
            return s
        for i in range(n):
            for j in range(l):
                if j%(2*n-2) == i or j%(2*n-2) == 2*n-2-i:
                    res_list.append(s[j])
        res=''.join(res_list)
        return res

Solution("ABCDEFGHIJKL",4)
Solution.convert(1,"ABCDEFGHIJKL",4)


'''
公众号
'''

#可视化

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
x=np.linspace(0,20) #linspace()函数制定横坐标 
plt.plot(x,.5+x) 
plt.plot(x,1+2*x,'--') 
plt.show()

df=pd.read_csv('gpu.csv')
df=df.groupby('year').aggregate(np.mean) 
years=df.index.values 
counts=df['trans_count'].values 
poly=np.polyfit(years,np.log(counts),deg=1) #polyfit（）表示用多项式拟合数据，deg=1表示多项式次 数为1 
print("Poly",poly) #输出拟合的系数 
plt.semilogy(years,counts,'o') #绘制对数坐标图 
plt.semilogy(years,np.exp(np.polyval(poly,years))) #polyval（）对多项式进行评估 
plt.show()

df=pd.read_csv('gpu.csv') 
df=df.groupby('year').aggregate(np.mean) 
gpu=pd.read_csv('gpu_transcount.csv') 
gpu=gpu.groupby('year').aggregate(np.mean) 
df=pd.merge(df,gpu,how='outer',left_index=True,right_index=True) #合并数据，并保留两个表的所有信 息 
df=df.replace(np.nan,0) #将缺失值或为0 
years=df.index.values 
counts=df['trans_count'].values 
gpu_counts=gpu['gpu_trans_count'].values 
cnt_log=np.log(counts) 
plt.scatter(years,cnt_log,c=200*years,
            s=20+200*gpu_counts/gpu_counts.max(),alpha=0.5) #c指定气泡 颜色，s设定气泡大小 
plt.show()

df=pd.read_csv('gpu.csv') 
df=df.groupby('year').aggregate(np.mean) 
gpu=pd.read_csv('gpu_transcount.csv') 
gpu=gpu.groupby('year').aggregate(np.mean) 
df=pd.merge(df,gpu,how='outer',left_index=True,right_index=True) 
df=df.replace(np.nan,0) 
years=df.index.values 
counts=df['trans_count'].values 
gpu_counts=gpu['gpu_trans_count'].values 
poly=np.polyfit(years,np.log(counts),deg=1) 
plt.plot(years,np.polyval(poly,years),label='Fit') 
gpu_start=gpu.index.values.min() 
y_ann=np.log(df.at[gpu_start,'trans_count']) 
ann_str="First GPU \n %d" % gpu_start #把gpu_start赋给ann_str并显示First GPU/n 
plt.annotate(ann_str,xy=(gpu_start,y_ann),arrowprops=dict(arrowstyle="->"),
             xytext=(-30,+70),textcoords="offset points") 
cnt_log=np.log(counts) 
plt.scatter(years,cnt_log,c=200*years,s=20+200*gpu_counts/gpu_counts.max(),
            alpha=0.5,label="Scatter Plot") 
plt.legend(loc='upper left') #显示图例，loc表示位置，即图例放在左上角 
plt.grid() #显示网格 
plt.xlabel("Year") #设置横轴标签 
plt.ylabel("Log Transistor Counts",fontsize=16) #设置纵轴标签 
plt.title("Moore's Law & Transistor Counts") #设置题目
plt.show()

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
df=pd.read_csv('gpu.csv') 
gpu=pd.read_csv('gpu_transcount.csv') 
df=df.groupby('year').aggregate(np.mean) 
df=pd.merge(df,gpu,how='outer',left_index=True,right_index=True)
df=df.replace(np.nan,0) 
df.plot(logy=True) 
df[df['trans_count']>0].plot(kind='scatter',x='trans_count',
  y='gpu_trans_count_x',loglog=True ) 
plt.show()

from pandas.tools.plotting import lag_plot 
df=pd.read_csv('gpu.csv') 
df=df.groupby('year').aggregate(np.mean) 
gpu=pd.read_csv('gpu_transcount.csv') 
gpu=gpu.groupby('year').aggregate(np.mean)
df=pd.merge(df,gpu,how='outer',left_index=True,right_index=True) 
df=df.replace(np.nan,0) 
lag_plot(np.log(df['trans_count'])) 
plt.show()

from pandas.tools.plotting import autocorrelation_plot 
df=pd.read_csv('gpu.csv') 
df=df.groupby('year').aggregate(np.mean) 
gpu=pd.read_csv('gpu_transcount.csv') 
gpu=gpu.groupby('year').aggregate(np.mean) 
df=pd.merge(df,gpu,how='outer',left_index=True,right_index=True) 
df=df.replace(np.nan,0) 
autocorrelation_plot(np.log(df['trans_count'])) #用自相关函数绘制自相关图 plt.show()


#交叉验证法
import pandas as pd 
import numpy as np 
df=pd.read_csv('wdbc.csv',header=None) 
df.head()

from sklearn.preprocessing import LabelEncoder 
X=df.loc[:,2:].values 
Y=df.loc[:,1].values 
le=LabelEncoder() #将字符串转换为整数 
Y=le.fit_transform(Y)

from sklearn import cross_validation 
X_train,X_test,Y_train,Y_test= cross_validation.train_test_split(X,Y,test_size=0.15,random_state=1)

from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline 
pipe_lr=Pipeline([('scl',StandardScaler()),('pca',PCA(n_components=2)),('clf',LogisticRegression(random_state=1))]) 
pipe_lr.fit(X_train,Y_train) 
print('Test Accuracy:%.3f'%pipe_lr.score(X_test,Y_test))

pipe_lr.fit

from sklearn.cross_validation import StratifiedKFold 
kfold=StratifiedKFold(Y_train,n_folds=10,random_state=1) 
scores=[] 
for k,(train,test) in enumerate(kfold): 
    pipe_lr.fit(X_train[train],Y_train[train]) 
    score=pipe_lr.score(X_train[test],Y_train[test]) 
    scores.append(score) 
    print('Fold:%s,Class dist.:%s,Acc:%.6f'%(k+1,np.bincount(Y_train[train]),score))

import matplotlib.pyplot as plt 
from sklearn.learning_curve import learning_curve 
pipe_lr=Pipeline([('scl',StandardScaler()),('clf',LogisticRegression(random_state=1))]) 
train_sizes,train_scores,test_scores=learning_curve(estimator=pipe_lr, X=X_train, y=Y_train, train_sizes=np.linspace(0.1,1,10), cv=10, n_jobs=1)

train_mean=np.mean(train_scores,axis=1) #axis=1，求行均值 
train_std=np.std(train_scores,axis=1) 
test_mean=np.mean(test_scores,axis=1) 
test_std=np.std(test_scores,axis=1) 
plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,label='training accuracy') #color：颜色；marker：形状；markersize：形状的大小；label：标签 
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue') #fill_between函数加入了平均准确率标准差的信息，用以表示评价结果的方差 
plt.plot(train_sizes,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='vali dation accuracy') 
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color='green') 
plt.grid() 
plt.xlabel('Number of training samples') 
plt.ylabel('Accuracy') 
plt.legend(loc='lower left')
plt.ylim([0.8,1.0]) 
plt.show()


'''
魔法命令 %run %timeit %time
'''

