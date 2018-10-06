# -*- coding: utf-8 -*-
"""
Created on Thu May 24 14:31:21 2018

@author: haoyu
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.plot([3,1,4,5,2])
plt.ylabel('grade')
plt.savefig('test',dpi=600)#存为像素点为600的图片
plt.show()

plt.plot([0,2,4,6,8],[3,1,4,5,2])
plt.ylabel('grade')
plt.axis([-1,10,0,6])
plt.show()

def f(t):
    return np.exp(-t)*np.cos(2*np.pi*t)

a=np.arange(0.0,5.0,0.02)
#图表位置
plt.subplot(2,2,1)
plt.plot(a,f(a))

plt.subplot(2,2,4)
plt.plot(a,np.cos(2*np.pi*a),'r--')
plt.show()

a=np.arange(10)
plt.plot(a,a*1.5,a,a*2.5,a,a*3.5,a,a*4.5)
plt.show()

#图表格式，线条格式
a=np.arange(10)
plt.plot(a,a*1.5,'go-',a,a*2.5,'rx',a,a*3.5,'*',a,a*4.5,'b-.')
plt.show()

#显示中文
matplotlib.rcParams['font.family']='SimHei'
plt.plot([3,1,4,5,2])
plt.ylabel('纵轴（值）')
plt.show()

matplotlib.rcParams['font.family']='STSong'#改变所有字体
matplotlib.rcParams['font.size']=20
a=np.arange(0.0,5.0,0.02)
plt.xlabel('横轴：时间')
plt.ylabel('纵轴：振幅')
plt.plot(a,np.cos(2*np.pi*a),'r--')
plt.show()

a=np.arange(0.0,5.0,0.02)
plt.xlabel('横轴：时间',fontproperties='SimHei',fontsize=20)#只改变坐标轴字体
plt.ylabel('纵轴：振幅',fontproperties='SimHei',fontsize=20)
plt.plot(a,np.cos(2*np.pi*a),'r--')
plt.show()

a=np.arange(0.0,5.0,0.02)
plt.plot(a,np.cos(2*np.pi*a),'r--')
plt.xlabel('横轴：时间',fontproperties='SimHei',fontsize=15,color='green')
plt.ylabel('纵轴：振幅',fontproperties='SimHei',fontsize=15)
plt.title(r'正弦波实例$y=cos(2\pi x)$',fontproperties='SimHei',fontsize=25)
plt.text(2,1,r'$\mu=100$',fontsize=15)
plt.axis([-1,6,-2,2])
plt.grid(True)#加入网格
plt.show()
#plt.annotate的用法（箭头）
a=np.arange(0.0,5.0,0.02)
plt.plot(a,np.cos(2*np.pi*a),'r--')
plt.xlabel('横轴：时间',fontproperties='SimHei',fontsize=15,color='green')
plt.ylabel('纵轴：振幅',fontproperties='SimHei',fontsize=15)
plt.title(r'正弦波实例$y=cos(2\pi x)$',fontproperties='SimHei',fontsize=25)
plt.annotate(r'$\mu=100$',xy=(2,1),xytext=(3,1.5),
             arrowprops=dict(facecolor='black',shrink=0.1,width=2))
plt.axis([-1,6,-2,2])
plt.grid(True)#加入网格
plt.show()


'''
pyplot的子绘图区域
plt.subplot2grid() 和 gridspec.GridSpec()
'''

##基础
import matplotlib as mpl
x=np.linspace(0,10,100)#平均切分
x
siny=np.sin(x)
plt.plot(x,siny)
plt.show()
cosy=np.cos(x)

plt.plot(x,siny)
plt.plot(x,cosy)
plt.show()

plt.plot(x,siny)
plt.plot(x,cosy)
plt.xlim(-5,15)
plt.ylim(0,1.5)
#或者plt.axis([-5,15,0,1.5])
plt.show()

plt.plot(x,siny)
plt.plot(x,cosy)
plt.axis([-5,15,0,1.5])
plt.xlabel('x axis')
plt.ylabel('y value')
plt.show()

#加图示
plt.plot(x,siny,label='sin(x)')
plt.plot(x,cosy,label='cos(x)')
plt.axis([-5,15,0,1.5])
plt.xlabel('x axis')
plt.ylabel('y value')
plt.legend()
plt.show()

###散点图scatter plot
plt.scatter(x,siny)
plt.scatter(x,cosy)
plt.show()

plt.scatter(np.random.normal(0,1,10000),np.random.normal(0,1,10000),alpha=0.1)#不透明度
plt.show()


