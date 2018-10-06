# -*- coding: utf-8 -*-
"""
Created on Tue May  1 19:12:38 2018

@author: haoyu
"""
#计算n!
def fact(n):
    s=1
    for i in range(1,n+1):
        s*=i
    return s#不被执行，调用才执行
a=fact(10)
print(a)

#七段数码管
#绘制日期
import turtle as t
def drawLine(draw):#绘制单段数码管
    t.pendown() if draw else t.penup()
    t.fd(40)
    t.right(90)
def drawDigit(digit):#根据数字绘制七段数码管
    drawLine(True) if digit in [2,3,4,5,6,8,9] else drawLine(False)
    drawLine(True) if digit in [0,1,3,4,5,6,7,8,9] else drawLine(False)
    drawLine(True) if digit in [0,2,3,5,6,8,9] else drawLine(False)
    drawLine(True) if digit in [0,2,6,8] else drawLine(False)
    t.left(90)
    drawLine(True) if digit in [0,4,5,6,8,9] else drawLine(False)
    drawLine(True) if digit in [0,2,3,5,6,7,8,9] else drawLine(False)
    drawLine(True) if digit in [0,1,2,3,4,7,8,9] else drawLine(False)
    t.left(180)
    t.penup()
    t.fd(20)
def drawDate(date):#获得要输出的数字
    for i in date:
        drawDigit(eval(i))
def main():
    t.setup(800,350,200,200)
    t.penup()
    t.fd(-300)
    t.pensize(5)
    drawDate('20181010')
    t.hideturtle()
    t.done()
main()

#绘制日期的扩展
import turtle as t
import time
def drawGap():
    t.penup()
    t.fd(5)
def drawLine(draw):#绘制单段数码管
    drawGap()
    t.pendown() if draw else t.penup()
    t.fd(40)
    drawGap()
    t.right(90)
def drawDigit(digit):#根据数字绘制七段数码管
    drawLine(True) if digit in [2,3,4,5,6,8,9] else drawLine(False)
    drawLine(True) if digit in [0,1,3,4,5,6,7,8,9] else drawLine(False)
    drawLine(True) if digit in [0,2,3,5,6,8,9] else drawLine(False)
    drawLine(True) if digit in [0,2,6,8] else drawLine(False)
    t.left(90)
    drawLine(True) if digit in [0,4,5,6,8,9] else drawLine(False)
    drawLine(True) if digit in [0,2,3,5,6,7,8,9] else drawLine(False)
    drawLine(True) if digit in [0,1,2,3,4,7,8,9] else drawLine(False)
    t.left(180)
    t.penup()
    t.fd(20)
def drawDate(date):#获得要输出的数字，格式为'%Y-%m=%d+'
    t.pencolor("red")
    for i in date:
        if i=='-':
            t.write('年',font=("Arial",18,"normal"))
            t.pencolor("green")
            t.fd(40)
        elif i=='=':
            t.write('月',font=("Arial",18,"normal"))
            t.pencolor("blue")
            t.fd(40)
        elif i=="+":
            t.write('日',font=("Arial",18,"normal"))
        else :
            drawDigit(eval(i))
def main():
    t.setup(800,350,200,200)
    t.penup()
    t.fd(-300)
    t.pensize(5)
    drawDate(time.strftime('%Y-%m=%d+',time.gmtime()))
    t.hideturtle()
    t.done()
main()

#递归的实现
def fact(n):
    if n==0:
        return 1
    else:
        return n*fact(n-1)

#字符串反转，递归过程
def rvs(s):
    if s=="":
        return s
    else:
        return rvs(s[1:])+s[0]

#斐波那契数列
#F(n)=F(n-1)+F(n-2)
def f(n):
    if n==1 or n==2:
        return 1
    else:
        return f(n-1)+f(n-2)

#汉诺塔模型，递归
count=0
def hanoi(n,src,dst,mid):#n=层数，src=原始柱子，dst=目标柱子，mid=中间柱子
    global count
    if n==1:
        print("{}:{}->{}".format(1,src,dst))
        count+=1
    else:
        hanoi(n-1,src,mid,dst)
        print("{}:{}->{}".format(n,src,dst))
        count+=1
        hanoi(n-1,mid,dst,src)
hanoi(4,"A","C","B")
print(count)
        
