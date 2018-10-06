# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 19:23:03 2018

@author: haoyu
"""
Temp=input("请输入带有符号的温度值：")
if Temp[-1]in['F','f']:
    C=(eval(Temp[0:-1])-32)/1.8
    print("转换后的温度是{:.2f}C".format(C))
elif Temp[-1]in['C','c']:
    F=1.8*eval(Temp[0:-1])+32
    print("转换后的温度是{:.2f}F".format(F))
else:print("输入格式错误")
