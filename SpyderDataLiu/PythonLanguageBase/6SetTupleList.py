# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:19:30 2018

@author: haoyu
"""
#集合
A={'p','y',123}
B=set('pypy123')
A-B
A&B
B-A
A|B
A^B

try:
    while True:
        print(A.pop(),end="")
except:
    pass
A
#数据去重
ls=['p','p','y','y',123]
s=set(ls)
lt=list(s)

#序列类型
ls=['python',123,'io']
ls[::-1]
s='python123.io'
s[::-1]
len(ls)
max(s)

creature='cat','dog','tiger','human'
creature
color=(0x001100,'blue',creature)
color
color[-1][2]
#列表
ls=['cat','dog','tiger',1023]
ls
lt=ls
lt
ls[1:2]=[1,2,3,4]
del ls[::3]
ls*2
ls.index(2)

#基本统计值
#CalStatistics1
def getNum():
    nums=[]
    iNumStr=input("请输入数字（回车退出）：")
    while iNumStr !="":
        nums.append(eval(iNumStr))
        iNumStr=input("请输入数字（回车退出）：")
    return nums
def mean(numbers):#计算均值
    s=0.0
    for num in numbers:
        s=s+num
    return s/len(numbers)
def dev(numbers,mean):#计算标准差
    sdev=0.0
    for num in numbers:
        sdev=sdev+(num-mean)**2
    return pow(sdev/(len(numbers)-1),0.5)
def median(numbers):#计算中位数
    sorted(numbers)#排序
    size=len(numbers)
    if size%2==0:
        med=(numbers[size//2-1]+numbers[size//2])/2
    else:
        med=numbers[size//2]
    return med
n=getNum()
m=mean(n)
print("平均值：{},方差：{:.2},中位数：{}.".format(m,dev(n,m),median(n)))

#键值对
d={'中国':'北京','美国':'华盛顿','法国':'巴黎'}
d
d['中国']
"中国"in d
d.keys()
d.values()
d.get('中国','伊斯兰堡')
d.get('巴基斯坦','伊斯兰堡')
items=list(d.items())
items=d.items()
