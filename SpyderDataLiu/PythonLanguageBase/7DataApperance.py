# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:31:42 2018

@author: haoyu
"""
#读入csv二维数据
fo=open(fname)
ls=[]
for i in fo:
    line=line.replace("\n","")
    ls.append(line.split(","))
fo.close()
#写入csv二维数据
ls=[[],[],[]]
f=open(fname,'w')
for item in ls:
    f.write(','.join(item)+'/n')
f.close()
#遍历二维数据所有元素
ls=[[],[],[]]
for row in ls:
    for column in row:
        print(ls[row][column])

#词云库
import wordcloud
c=wordcloud.WordCloud()
c.generate("wordcloud by Python")
c.to_file("pywordcloud.png")

#GovRptWordCloud1
import jieba
import wordcloud
f=open("新时代中国特色社会主义.txt","r",encoding="utf-8")
t=f.read()
f.close()
ls=jieba.lcut(t)
txt=" ".join(ls)
w=wordcloud.WordCloud(font_path='msyh.ttc,width=1000,height=700,\
                      background_color="white")
w.generate(txt)
w.to_file("grwordcloud.png")
#背景图改变
import jieba
import wordcloud
from scipy.misc import imread#加载库
mask=imread("fivestart.png")#加载图
f=open("新时代中国特色社会主义.txt","r",encoding="utf-8")
t=f.read()
f.close()
ls=jieba.lcut(t)
txt=" ".join(ls)
w=wordcloud.WordCloud(font_path='msyh.ttc,width=1000,height=700,####mask=mask\
                      background_color="white")
w.generate(txt)
w.to_file("grwordcloud.png")

