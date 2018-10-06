# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 22:26:46 2018

@author: haoyu
"""
import time
time.time()
time.ctime()
time.gmtime()

t=time.gmtime()
time.strftime("%Y-%m-%d %H:%M:%S",t)

#文本进度条
#TextProBar1
import time
scale=10
print("------执行开始------")
for i in range(scale+1):
    a='*'*i
    b='.'*(scale-i)
    c=(i/scale)*100
    print("{:^3.0f}%[{}->{}]".format(c,a,b))
    time.sleep(0.1)
print("------执行结束------")

#TextProBar2
import time
for i in range(101):
    print("\r{:3}%".format(i),end="")
    time.sleep(0.1)
    
#TextProBar3
import time
scale=50
print("执行开始".center(scale//2,"-"))
start=time.perf_counter()
for i in range(scale+1):
    a='*'*i
    b='.'*(scale-i)
    c=(i/scale)*100
    dur=time.perf_counter()-start
    print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c,a,b,dur),end='')
    time.sleep(0.1)
print("\n"+"执行结束".center(scale//2,'-'))