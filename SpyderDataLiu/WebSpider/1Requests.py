# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 10:36:45 2018

@author: haoyu
"""

import requests

r=requests.get('http://www.baidu.com')
r.status_code
r.encoding
r.apparent_encoding
r.encoding = 'utf-8'
r.text
type(r)

'''
京东商品爬取
url:http://item.jd.com/2967929.html
'''
r=requests.get('http://item.jd.com/2967929.html')
r.status_code
r.encoding
r.text[:1000]

#全代码
import requests
url='http://item.jd.com/2967929.html'
try:
    r=requests.get(url)
    r.raise_for_status()
    r.encoding=r.apparent_encoding
    print(r.text[:1000])
except:
    print('爬取失败')
    
'''
亚马逊商品爬取   
url:https://www.amazon.cn/gp/product/B01M8L5Z3Y 
'''
r=requests.get('https://www.amazon.cn/gp/product/B01M8L5Z3Y ')
r.status_code
r.encoding
r.encoding=r.apparent_encoding
r.text

r.request.headers

kv={'user-agent':'Mozilla/5.0'}
url='https://www.amazon.cn/gp/product/B01M8L5Z3Y'
r=requests.get(url,headers=kv)
r.status_code
r.request.headers
r.text[1000:2000]

#全代码
import requests
url='https://www.amazon.cn/gp/product/B01M8L5Z3Y'
try:
    kv={'user-agent':'Mozilla/5.0'}
    r=requests.get(url,headers=kv)
    r.raise_for_status()
    r.encoding=r.apparent_encoding
    print(r.text[1000:2000])
except:
    print('爬取失败')

'''
百度360搜索关键词提交
url:http://www.baidu.com/s?wd=##keyword
url:http://www.so.com/s?q=##keyword
'''
kv={'wd':'Python'}
r=requests.get('http://www.baidu.com/s',params=kv)
r.status_code
r.request.url
len(r.text)

#全代码
import requests
keyword='Python'
try:
    kv={'wd':keyword}
    r=requests.get('http://www.baidu.com/s',params=kv)
    print(r.request.url)
    r.raise_for_status()
    print(len(r.text))
except:
    print('爬取失败')

'''
网络图片的爬取和存储
网络图片链接的格式：http://www.example.com/picture.jpg
'''
path='D:\\Anaconda3\\SpyderDataLiu\\Python网络爬虫与信息提取\\a.jpg'
url='http://image.nationalgeographic.com.cn/2017/0211/20170211061910157.jpg'
r=requests.get(url)
r.status_code
with open(path,'wb') as f:
    f.write(r.content)
f.close()

'''
IP地址归属地自动查询

'''
url='http://m.ip138.com/ip.asp?ip='
r=requests.get(url+'202.204.80.112')
r.status_code
r.text[-500:]



