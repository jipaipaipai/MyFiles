# -*- coding: utf-8 -*-
"""
Created on Fri May  4 09:07:58 2018

@author: haoyu
"""
#文本形式打开
tf=open("f.txt","rt")
print(tf.readline())
tf.close()
#二进制形式打开
bf=open("f.txt","rb")
print(bf.readline())
bf.close()

#文件打开
f=open("f.txt")
f.close()

#遍历全文本，方法1
fname=input("请输入要打开的文件名称：")
fo=open(fname,"r")
txt=fo.read()
#........
#........对全文txt进行处理
fo.close()

#遍历全文本，方法2
fname=input("请输入要打开的文件名称：")
fo=open(fname,"r")
txt=fo.read(2)
while txt !="":
    #......
    #对txt进行处理
    txt=fo.read(2)
fo.close()

#逐行遍历文件，方法1
fname=input("请输入要打开的文件名称：")
fo=open(fname,"r")
for line in fo.readlines():
    print(line)
fo.close()
#逐行遍历文件，方法2
fname=input("请输入要打开的文件名称：")
fo=open(fname,"r")
for line in fo:
    print(line)
fo.close()

#文件写入
fo=open("output.txt","w+")
ls=['中国','法国','美国']
fo.writelines(ls)
fo.seek(0)
for line in fo:
    print(line)
fo.close()

