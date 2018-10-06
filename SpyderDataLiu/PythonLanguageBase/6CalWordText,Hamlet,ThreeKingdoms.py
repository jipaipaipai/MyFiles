# -*- coding: utf-8 -*-
"""
Created on Wed May  2 19:33:02 2018

@author: haoyu
"""

#词频分析
#CalHamlet1
def getText():
    txt=open("hamlet.txt","r").read()
    txt=txt.lower()
    for ch in '!"#$%&()*+,-./:;<=>?@[\\]^_‘{|}~':
        txt=txt.replace(ch," ")
    return txt
hamletTxt=getText()
words=hamletTxt.split()
counts={}
for word in words:
    counts[word]=counts.get(word,0)+1
items=list(counts.items())
items.sort(key=lambda x:x[1],reverse=True)
for i in range(10):
    word,count=items[i]
    print("{0:<10}{1:>5}".format(word,count))

#CalThreeKingdoms1
import jieba
txt=open("threekingdoms.txt","r",encoding="utf-8").read()
words=jieba.lcut(txt)
counts={}
for word in words:
    if len(word)==1:
        continue
    else:
        counts[word]=counts.get(word,0)+1
items=list(counts.items())
items.sort(key=lambda x:x[1],reverse=True)
for i in range(15):
    word,count=items[i]
    print("{0:<10}{1:>5}".format(word,count))
