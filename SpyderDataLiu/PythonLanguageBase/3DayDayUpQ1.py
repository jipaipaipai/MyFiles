# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 20:53:02 2018

@author: haoyu
"""
#DayDayUpQ1.py
pow(2,3)
round(0.125,2)
10/3
10//3#整数除
10%3#求余

#DayDayUpQ2
dayfactor=0.005
dayup=pow(1+dayfactor,365)
daydown=pow(1-dayfactor,365)

#DayDayUpQ3  双休
dayup=1.0
dacfactor=0.01
for i in range(365):
    if i % 7 in [6,0]:
        dayup= dayup*(1-dayfactor)
    else:
        dayup=dayup*(1+dayfactor)
print("工作日的力量：{:.2f}".format(dayup))

#DayDayUpQ4
def dayUP(df):
    dayup=1
    for i in range(365):
        if i % 7 in [6,0]:
            dayup=dayup*(1-0.01)
        else:
            dayup=dayup*(1+df)
    return dayup
dayfactor=0.01
while dayUP(dayfactor)<37.78:
    dayfactor+=0.001
print("工作日努力的参数是:{:.3f}".format(dayfactor))

#WeekName
weekStr="星期一星期二星期三星期四星期五星期六星期日"
weekId=eval(input("请输入星期数字（1-7）："))
pos=(weekId-1)*3
print(weekStr[pos:pos+3])

weekStr="一二三四五六日"
weekId=eval(input("请输入星期数字"))
print("星期"+weekStr[weekId-1])

