# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 10:37:37 2018

@author: haoyu
"""

'''
正则表达式
'''
#re.search函数
import re
match=re.search(r'[1-9]\d{5}','BIT 100081')
if match:
    print(match.group(0))

#re.match函数
match=re.match(r'[1-9]\d{5}','BIT 100081')
if match:
    match.group(0)
    
'''
淘宝商品信息比价定向爬虫
'''    
import requests
import re

def getHTMLText(url):
     try:
        r=requests.get(url,timeout=30)
        r.raise_for_status()
        r.encoding=r.apparent_encoding
        return r.text
     except:
        return ''

def parsePage(ilt,html):
    try:
        plt=re.findall(r'\"view_price\"\:\"[\d\.]*\"',html)
        tlt=re.findall(r'\"raw_title\"\:\".*?\"',html)
        for i in range(len(plt)):
            price=eval(plt[i].split(':')[1])
            title=eval(tlt[i].split(':')[1])
            ilt.append([price,title])
    except:
        print('')
    
def printGoodsList(ilt):
    tplt="{:4}\t{:8}\t{:16}"
    print(tplt.format('序号','价格','商品名称'))
    count=0
    for g in ilt:
        count+=1
        print(tplt.format(count,g[0],g[1]))
    
def main():
    goods='书包'
    depth=2
    start_url='https://s.taobao.com/search?q='+goods
    infoList=[]
    for i in range(depth):
        try:
            url=start_url+'&s='+str(44*i)
            html=getHTMLText(url)
            parsePage(infoList,html)
        except:
            continue
        printGoodsList(infoList)

main()

'''
股票数据定向爬虫
'''
import requests
import re
from bs4 import BeautifulSoup
import traceback

def getHTMLText(url):
     try:
        r=requests.get(url,timeout=30)
        r.raise_for_status()
        r.encoding=r.apparent_encoding
        return r.text
     except:
        return ''

def getStockList(lst,stockURL):
    html=getHTMLText(stockURL)
    soup=BeautifulSoup(html,'html.parser')
    a=soup.find_all('a')
    for i in a:
        try:
            href=i.attrs['href']
            lst.append(re.findall(r'[s][hz]\d{6}',href)[0])
        except:
            continue

def getStockInfo(lst,stockURL,fpath):
    for stock in lst:
        url=stockURL+stock+'.html'
        html=getHTMLText(url)
        try:
            if html=='':
                continue
            infoDict={}
            soup=BeautifulSoup(html,'html.parser')
            stockInfo=soup.find('div',attrs={'class':'stock-bets'})
            
            name=stockInfo.find_all(attrs={'class':'bets-name'})[0]
            infoDict.update({'股票名称':name.text.spilt()[0]})
            
            keyList=stockInfo.find_all('dt')
            valueList=stockInfo.find_all('dd')
            for i in range(len(keyList)):
                key=keyList[i].text
                val=valueList[i].text
                infoDict[key]=val
                
            with open(fpath,'a',encoding='utf-8') as f:
                f.write(str(infoDict)+'\n')
        except:
            traceback.print_exc()
            continue

def main():
    stock_list_url='http://quote.eastmoney.com/stocklis.html'
    stock_info_url='https://gupiao.baidu.com/stock/'
    output_file='D:/Anaconda3/SpyderDataLiu/Python网络爬虫与信息提取/price.txt'
    slist=[]
    getStockList(slist,stock_list_url)
    getStockInfo(slist,stock_info_url,output_file)
    
main()




