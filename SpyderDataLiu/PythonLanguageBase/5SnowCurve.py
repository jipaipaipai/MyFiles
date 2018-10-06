# -*- coding: utf-8 -*-
"""
Created on Tue May  1 21:28:49 2018

@author: haoyu
"""
#科赫曲线，雪花曲线
import turtle as t
def koch(size,n):
    if n==0:
        t.fd(size)
    else:
        for angle in [0,60,-120,60]:
            t.left(angle)
            koch(size/3,n-1)
def main():
    t.setup(800,400)
    t.penup()
    t.goto(-300,-50)
    t.pendown()
    t.pensize(2)
    koch(600,3)#3阶科赫曲线
    t.hideturtle()
    t.done()
main()

#科赫曲线进一步到科赫雪花
import turtle as t
def koch(size,n):
    if n==0:
        t.fd(size)
    else:
        for angle in [0,60,-120,60]:
            t.left(angle)
            koch(size/3,n-1)
def main():
    t.setup(600,600)
    t.penup()
    t.goto(-200,100)
    t.pendown()
    t.pensize(2)
    level=3
    koch(400,level)
    t.right(120)
    koch(400,level)
    t.right(120)
    koch(400,level)
    t.hideturtle()
    t.done()
main()