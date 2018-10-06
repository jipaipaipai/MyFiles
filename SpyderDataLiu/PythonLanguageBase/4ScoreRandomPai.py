# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 18:18:06 2018

@author: haoyu
"""
score=eval(input("输入分数"))
if score>=60 and score<70:
    grade="D"
elif score>=70 and score<80:
    grade="C"
elif score>=80 and score<90:
    grade="B"
elif score>=90:
    grade="A"
print("输入成绩属于级别{}".format(grade))

try:
    num=eval(input("请输入整数"))
    print(num**2)
except:
    print("输入不是整数")
    
#BMI1
height,weight=eval(input("请输入身高（米）和体重\（公斤）[逗号隔开]："))
bmi=weight/pow(height,2)
print("BMI数值为：{:.2f}".format(bmi))
who,nat="",""
if bmi<18.5:
    who,nat="thin","thin"
elif 18.5<=bmi<24:
    who,nat="nor","nor"
elif 24<=bmi<25:
    who,nat="nor","fat"
elif 25<=bmi<28:
    who,nat="fat","fat"
elif 28<=bmi<30:
    who,nat="fat","obesity"
else:
    who,nat="obesity","obesity"
print("BMI指标为，国际'{0}',国内'{1}'".format(who,nat))

#遍历循环
for i in range(5):
    print(i)
    
for i in range(1,6):
    print(i)
for i in range(1,6,2):
    print(i)
    
for c in "Python123":
    print(c,end=",")
    
for item in [123,"PY",456]:
    print(item,end=",")

#无限循环,条件循环
a=3
while a>0:
    a=a-1
    print(a)
    
for c in "PYTHON":
    if c=="T":
        continue
    print(c,end="")

for c in "PYTHON":
    if c=="T":
        break
    print(c,end="")

s="PYTHON"
while s !="":
    for c in s:
        print(c,end="")
    s=s[:-1]

s="PYTHON"
while s !="":
    for c in s:
        if c =="t":
            break
        print(c,end="")
    s=s[:-1]
    
#random库
import random
random.seed(10)
random.random()

random.randint(10,100)
random.randrange(10,100,10)
random.getrandbits(16)
random.uniform(10,100)
random.choice([1,2,3,4,5,6,7,8,9])
s=[1,2,3,4,5,6,7,8,9];random.shuffle(s);print(s)

#圆周率
#公式法
pi=0
N=100
for k in range(N):
    pi+=1/pow(16,k)*(\
             4/(8*k+1)-2/(8*k+4)-\
             1/(8*k+5)-1/(8*k+6))
print("圆周率值是：{}".format(pi))
#撒点法
from random import random
from time import perf_counter
DRATS=1000*1000
hits=0.0
start=perf_counter()
for i in range(1,DRATS+1):
    x,y=random(),random()
    dist=pow(x**2+y**2,0.5)
    if dist<=1.0:
        hits=hits+1
pi=4*(hits/DRATS)
print("圆周率值是：{}".format(pi))
print("运行时间是：{:.5f}s".format(perf_counter()-start))


