# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 14:35:13 2018

@author: haoyu
"""


#安装小测:http://python123.io/ws/demo.html
from bs4 import BeautifulSoup
import requests

r=requests.get('http://python123.io/ws/demo.html')
r.text
demo=r.text

soup=BeautifulSoup(demo,'html.parser')
print(soup.prettify())
soup.title#获取标题
tag=soup.a#获取标签
tag

soup.a.name
soup.a.parent.name#上一级名字
soup.a.parent.parent.name#上上级名字

tag.attrs#获取属性
tag.attrs['class']
type(tag.attrs)
type(tag)

soup.a
soup.a.string

'''
用bs4库的html内容遍历方法
'''
soup.head
soup.head.contents
soup.body.contents
len(soup.body.contents)
soup.body.contents[1]

soup.title.parent
soup.html.parent

soup.a.next_sibling
soup.a.previous_sibling

'''
基于bs4库的html格式化和编码
'''
soup.prettify()
print(soup.prettify())

'''
信息提取的一般方法
'''
#实例
#提取HTML中所有的URL链接
for link in soup.find_all('a'):
    print(link.get('href'))
'''
基于bs4库的HTML内容查找方法
'''
#<>.find_all()
soup.find_all('a')
soup('a')#二者等价
soup.find_all(['a','b'])

for tag in soup.find_all(True):
    print(tag.name)

import re
for tag in soup.find_all(re.compile('b')):
    print(tag.name)

'''
实例
中国大学排名定向爬虫
'''
import requests
import bs4
from bs4 import BeautifulSoup

def getHTMLText(url):
    try:
        r=requests.get(url,timeout=30)
        r.raise_for_status()
        r.encoding=r.apparent_encoding
        return r.text
    except:
        return ''

def fillUnivList(ulist,html):
    soup=BeautifulSoup(html,'html.parser')
    for tr in soup.find('tbody').children:
        if isinstance(tr,bs4.element.Tag):
            tds=tr('td')
            ulist.append([tds[0].string,tds[1].string,tds[3].string])

def printUnivList(ulist,num):
    print('{:^10}\t{:^6}\t{:^10}'.format('排名','学校名称','总分'))
    for i in range(num):
        u=ulist[i]
        print('{:^10}\t{:^6}\t{:^10}'.format(u[0],u[1],u[2]))    

def main():
    uinfo=[]
    url='http://www.zuihaodaxue.cn/zuihaodaxuepaiming2016.html'
    html=getHTMLText(url)
    fillUnivList(uinfo,html)
    printUnivList(uinfo,20)

main()

'''
实例
代码优化
'''
import requests
import bs4
from bs4 import BeautifulSoup

def getHTMLText(url):
    try:
        r=requests.get(url,timeout=30)
        r.raise_for_status()
        r.encoding=r.apparent_encoding
        return r.text
    except:
        return ''

def fillUnivList(ulist,html):
    soup=BeautifulSoup(html,'html.parser')
    for tr in soup.find('tbody').children:
        if isinstance(tr,bs4.element.Tag):
            tds=tr('td')
            ulist.append([tds[0].string,tds[1].string,tds[3].string])

def printUnivList(ulist,num):
    tplt='{0:^10}\t{1:{3}^10}\t{2:^10}'#输出格式化
    print(tplt.format('排名','学校名称','总分',chr(12288)))#中文空格的字符代码
    for i in range(num):
        u=ulist[i]
        print(tplt.format(u[0],u[1],u[2],chr(12288)))    

def main():
    uinfo=[]
    url='http://www.zuihaodaxue.cn/zuihaodaxuepaiming2016.html'
    html=getHTMLText(url)
    fillUnivList(uinfo,html)
    printUnivList(uinfo,20)

main()



