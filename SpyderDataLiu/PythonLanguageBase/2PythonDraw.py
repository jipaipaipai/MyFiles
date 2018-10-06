# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 19:33:15 2018

@author: haoyu
"""
#PythonDraw.py
import turtle
turtle.setup(650,350,200,200)
turtle.penup()
turtle.fd(-250)
turtle.pendown()
turtle.pensize(25)
turtle.pencolor("purple")
turtle.seth(-40)
for i in range(4):
    turtle.circle(40,80)
    turtle.circle(-40,80)
turtle.circle(40,80/2)
turtle.fd(40)
turtle.circle(16,180)
turtle.fd(40*2/3)
turtle.done()

#五角星
import turtle as t
t.setup(600,300)
t.penup()
t.fd(-100)
t.pendown()
t.pensize(10)
t.pencolor("red")
t.fd(80)
t.seth(60)
t.fd(80)
t.seth(-60)
t.fd(80)
t.seth(0)
t.fd(80)
t.seth(-150)
t.fd(80)
t.seth(-60)
t.fd(80)
t.seth(150)
t.fd(90)
t.seth(-150)
t.fd(90)
t.seth(60)
t.fd(80)
t.seth(150)
t.fd(80)
t.done()

#爱心
import turtle as t
t.setup(800,400)
t.penup()
t.goto(-100,50)
t.pendown()
t.pencolor("red")
t.pensize(15)
t.seth(90)
t.circle(-50,180)
t.seth(90)
t.circle(-50,180)
t.seth(-120)
t.fd(200)
t.seth(120)
t.fd(200)
t.penup()
t.goto(100,-20)
t.pendown()
t.seth(-90)
t.fd(100)
t.seth(0)
t.fd(50)
t.done()
