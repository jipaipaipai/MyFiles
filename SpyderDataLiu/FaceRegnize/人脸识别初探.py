# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:09:43 2018

@author: haoyu
"""

'''
方法1
'''
import cv2
image = cv2.imread(r'D:\Anaconda3\SpyderDataLiu\PicRecgnize\IMG_0597.jpg')
color=(0,255,0)
#i=1

grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

classfier = cv2.CascadeClassifier(r"D:\Anaconda3\SpyderDataLiu\PicRecgnize\haarcascades\haarcascade_frontalface_default.xml")

faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
if len(faceRects) > 0:  # 大于0则检测到人脸
    for faceRect in faceRects:  # 单独框出每一张人脸
        x, y, w, h = faceRect
        cv2.rectangle(image, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 10) #5控制绿色框的粗细
 
# 写入图像
cv2.namedWindow('1',0)
cv2.imshow('1',image)
cv2.waitKey(0)

#cv2.imwrite(str(i)+'.jpg',image)


'''
方法2，dlib库
'''
#cap=cv2.VideoCapture(0)
#cap.set(3,480)
#cap.isOpened()
#flag, im_rd = cap.read()
#cap.release()



import dlib
from skimage import io

# 使用Dlib的正面人脸检测器frontal_face_detector
detector = dlib.get_frontal_face_detector()

# 图片所在路径
path_pic = 'D:/Anaconda3/SpyderDataLiu/PicRecgnize/'
img = io.imread(path_pic+"IMG_0597.jpg")
 
# 生成dlib的图像窗口
win = dlib.image_window()
win.set_image(img)

# 使用detector检测器来检测图像中的人脸
dets = detector(img, 1)
print("人脸数：", len(dets))
 
for i, d in enumerate(dets):
    print("第", i+1, "个人脸的矩形框坐标：",
          "left:", d.left(), "right:", d.right(), "top:", d.top(), "bottom:", d.bottom())


# 绘制矩阵轮廓
win.add_overlay(dets)

# 保持图像
dlib.hit_enter_to_continue()



'''
方法3，绘制68特征点
'''

import cv2
import dlib
from skimage import io

# 使用Dlib的正面人脸检测器frontal_face_detector
detector = dlib.get_frontal_face_detector()

# dlib的68点模型
path_68points_model = "D:/Anaconda3/SpyderDataLiu/PicRecgnize/dat/"
predictor = dlib.shape_predictor(path_68points_model+"shape_predictor_68_face_landmarks.dat")

# 图片所在路径
path_pic = 'D:/Anaconda3/SpyderDataLiu/PicRecgnize/'
img = io.imread(path_pic+"IMG_0597.jpg")

# 生成dlib的图像窗口
win = dlib.image_window()
win.set_image(img)

# 使用detector检测器来检测图像中的人脸
dets = detector(img, 1)
print("人脸数：", len(dets))

for i, d in enumerate(dets):
    print("第", i+1, "个人脸的矩形框坐标：",
          "left:", d.left(), "right:", d.right(), "top:", d.top(), "bottom:", d.bottom())

    # 使用predictor来计算面部轮廓
    shape = predictor(img, dets[i])
    # 绘制面部轮廓
    win.add_overlay(shape)

# 绘制矩阵轮廓
win.add_overlay(dets)

# 保持图像
dlib.hit_enter_to_continue()



'''
方法4，特征点数字标记
'''

import dlib                     #人脸识别的库dlib
import numpy as np              #数据处理的库numpy
import cv2                      #图像处理的库OpenCv

# dlib预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/Anaconda3/SpyderDataLiu/PicRecgnize/dat/shape_predictor_68_face_landmarks.dat')

path="D:/Anaconda3/SpyderDataLiu/PicRecgnize/"

# cv2读取图像
img=cv2.imread(path+"IDPic.jpg")

# 取灰度
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 人脸数rects
rects = detector(img_gray, 0)

for i in range(len(rects)):
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])

    for idx, point in enumerate(landmarks):
        # 68点的坐标
        pos = (point[0, 0], point[0, 1])

        # 利用cv2.circle给每个特征点画一个圈，共68个
        cv2.circle(img, pos, 5, color=(0, 255, 0))

        # 利用cv2.putText输出1-68
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(idx+1), pos, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

cv2.namedWindow("img", 0)
cv2.imshow("img", img)
cv2.waitKey(0)

#cv2.imwrite('1.jpg',img)



'''
方法5，摄像头特征点标定
'''
# 调用摄像头，进行人脸捕获，和68个特征点的追踪

import dlib         # 人脸识别的库dlib
import numpy as np  # 数据处理的库numpy
import cv2          # 图像处理的库OpenCv
import time

# 储存截图的目录
path_screenshots = "D:/Anaconda3/SpyderDataLiu/PicRecgnize/ScreenShot/"

# dlib预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/Anaconda3/SpyderDataLiu/PicRecgnize/dat/shape_predictor_68_face_landmarks.dat')

# 创建cv2摄像头对象
cap = cv2.VideoCapture(0)

# cap.set(propId, value)
# 设置视频参数，propId设置的视频参数，value设置的参数值
cap.set(3, 480)

# 截图screenshoot的计数器
cnt = 0

# cap.isOpened（） 返回true/false 检查初始化是否成功
while (cap.isOpened()):

    # cap.read()
    # 返回两个值：
    #    一个布尔值true/false，用来判断读取视频是否成功/是否到视频末尾
    #    图像对象，图像的三维矩阵
    flag, im_rd = cap.read()

    # 每帧数据延时1ms，延时为0读取的是静态帧
    k = cv2.waitKey(1)

    # 取灰度
    img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)

    # 人脸数rects
    dets = detector(img_gray, 0)

    # print(len(dets))

    # 待会要写的字体
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 标68个点
    if len(dets) != 0:
        # 检测到人脸
        for i in range(len(dets)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(im_rd, dets[i]).parts()])

            for idx, point in enumerate(landmarks):
                # 68点的坐标
                pos = (point[0, 0], point[0, 1])

                # 利用cv2.circle给每个特征点画一个圈，共68个
                cv2.circle(im_rd, pos, 2, color=(139, 0, 0))

                # 利用cv2.putText输出1-68
                cv2.putText(im_rd, str(idx + 1), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(im_rd, "faces: " + str(len(dets)), (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        # 没有检测到人脸
        cv2.putText(im_rd, "no face", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

    # 添加说明
    im_rd = cv2.putText(im_rd, "s: screenshot", (20, 400), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    im_rd = cv2.putText(im_rd, "q: quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # 按下s键保存
    if k == ord('s'):
        cnt += 1
        print(path_screenshots+ "screenshoot" + "_" + str(cnt) + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".jpg")
        cv2.imwrite(path_screenshots+ "screenshoot" + "_" + str(cnt) + "_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".jpg", im_rd)

    # 按下q键退出
    if k == ord('q'):
        break

    # 窗口显示

    # 参数取0可以拖动缩放窗口，为1不可以
    cv2.namedWindow("camera", 0)
    #cv2.namedWindow("camera", 1)

    cv2.imshow("camera", im_rd)
    

# 释放摄像头
cap.release()

# 删除建立的窗口
cv2.destroyAllWindows()



'''
方法6，摄像头人脸识别
'''

import dlib         # 人脸识别的库dlib
import numpy as np  # 数据处理的库numpy
import cv2          # 图像处理的库OpenCv

# dlib预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/Anaconda3/SpyderDataLiu/PicRecgnize/dat/shape_predictor_68_face_landmarks.dat')

# 创建cv2摄像头对象
cap = cv2.VideoCapture(0)

# cap.set(propId, value)
# 设置视频参数，propId设置的视频参数，value设置的参数值
cap.set(3, 480)

# 截图screenshoot的计数器
cnt_ss = 0

# 人脸截图的计数器
cnt_p = 0

# 保存
path_save = "D:/Anaconda3/SpyderDataLiu/PicRecgnize/ScreenShot/"

# cap.isOpened（） 返回true/false 检查初始化是否成功
while cap.isOpened():

    # cap.read()
    # 返回两个值：
    #    一个布尔值true/false，用来判断读取视频是否成功/是否到视频末尾
    #    图像对象，图像的三维矩阵q
    flag, im_rd = cap.read()

    # 每帧数据延时1ms，延时为0读取的是静态帧
    kk = cv2.waitKey(1)

    # 取灰度
    img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)

    # 人脸数rects
    rects = detector(img_gray, 0)

    # print(len(rects))

    # 待会要写的字体
    font = cv2.FONT_HERSHEY_SIMPLEX

    if len(rects) != 0:
        # 检测到人脸

        # 矩形框
        for k, d in enumerate(rects):

            # 计算矩形大小
            # (x,y), (宽度width, 高度height)
            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])

            # 计算矩形框大小
            height = d.bottom() - d.top()
            width = d.right() - d.left()

            # 根据人脸大小生成空的图像
            cv2.rectangle(im_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)
            im_blank = np.zeros((height, width, 3), np.uint8)

            # 按下's'保存摄像头中的人脸到本地
            if kk == ord('s'):
                cnt_p += 1
                for ii in range(height):
                    for jj in range(width):
                        im_blank[ii][jj] = im_rd[d.top() + ii][d.left() + jj]
                # 存储人脸图像文件
                cv2.imwrite(path_save + "img_face_" + str(cnt_p) + ".jpg", im_blank)
                print("写入本地：", path_save + "img_face_" + str(cnt_p) + ".jpg")

        # 显示人脸数
        cv2.putText(im_rd, "faces: " + str(len(rects)), (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    else:
        # 没有检测到人脸
        cv2.putText(im_rd, "no face", (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    # 添加说明
    im_rd = cv2.putText(im_rd, "s: save face", (20, 400), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    im_rd = cv2.putText(im_rd, "q: quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # 按下q键退出
    if kk == ord('q'):
        break

    # 窗口显示
    # cv2.namedWindow("camera", 0) # 如果需要摄像头窗口大小可调
    cv2.imshow("camera", im_rd)

# 释放摄像头
cap.release()

# 删除建立的窗口
cv2.destroyAllWindows()


#   return_128d_features()          获取某张图像的128d特征
#   write_into_csv()                将某个文件夹中的图像读取特征兵写入csv
#   compute_the_mean()              从csv中读取128d特征，并计算特征均值

import cv2
import os
import dlib
from skimage import io
import csv
import numpy as np
import pandas as pd

path_pics = "D:/Anaconda3/SpyderDataLiu/PicRecgnize/ScreenShot/"
path_csv = "D:/Anaconda3/SpyderDataLiu/PicRecgnize/csv/"

# detector to find the faces
detector = dlib.get_frontal_face_detector()

# shape predictor to find the face landmarks
predictor = dlib.shape_predictor("D:/Anaconda3/SpyderDataLiu/PicRecgnize/dat/shape_predictor_5_face_landmarks.dat")

# face recognition model, the object maps human faces into 128D vectors
facerec = dlib.face_recognition_model_v1("D:/Anaconda3/SpyderDataLiu/PicRecgnize/dat/dlib_face_recognition_resnet_model_v1.dat")

# 返回单张图像的128D特征
def return_128d_features(path_img):
    img = io.imread(path_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img_gray, 1)

    if(len(dets)!=0):
        shape = predictor(img_gray, dets[0])
        face_descriptor = facerec.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
        print("no face")

   # print(face_descriptor)
    return face_descriptor

#return_128d_features(path_pics+"img_face_13.jpg")

# 将文件夹中照片特征提取出来，写入 csv
# 输入 input:
#   path_pics:  图像文件夹的路径
#   path_csv:   要生成的 csv 路径

def write_into_csv(path_pics ,path_csv):
    dir_pics = os.listdir(path_pics)

    with open(path_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(dir_pics)):
            # 调用return_128d_features()得到128d特征
            print(path_pics+dir_pics[i])
            features_128d = return_128d_features(path_pics+dir_pics[i])
          #  print(features_128d)
            # 遇到没有检测出人脸的图片跳过
            if features_128d==0:
                i += 1
            else:
                writer.writerow(features_128d)

#write_into_csv(path_pics, path_csv+"default_person.csv")

path_csv_rd = "D:/Anaconda3/SpyderDataLiu/PicRecgnize/csv/default_person.csv"

# 从csv中读取数据，计算128d特征的均值
def compute_the_mean(path_csv_rd):
    column_names = []

    for i in range(128):
        column_names.append("features_" + str(i + 1))

    rd = pd.read_csv(path_csv_rd, names=column_names)

    # 存放128维特征的均值
    feature_mean = []

    for i in range(128):
        tmp_arr = rd["features_"+str(i+1)]
        tmp_arr = np.array(tmp_arr)

        # 计算某一个特征的均值
        tmp_mean = np.mean(tmp_arr)

        feature_mean.append(tmp_mean)

    print(feature_mean)
    return feature_mean

compute_the_mean(path_csv_rd)


#欧式距离对比
import dlib         # 人脸识别的库dlib
import numpy as np  # 数据处理的库numpy
import cv2          # 图像处理的库OpenCv

# face recognition model, the object maps human faces into 128D vectors
facerec = dlib.face_recognition_model_v1("D:/Anaconda3/SpyderDataLiu/PicRecgnize/dat/dlib_face_recognition_resnet_model_v1.dat")

# 计算两个向量间的欧式距离
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    print(dist)

    if dist > 0.4:
        return "diff"
    else:
        return "same"

features_mean_default_person = [-0.06440531994615283, 0.19109050716672624, 0.028835981818182126,
                                -0.046459752800209184, -0.05560113862156868, -0.05298282179449286,
                                -0.07783002140266555, -0.12995321729353496, 0.11059773181165967,
                                -0.05596338851111276, 0.2598048171826771, -0.02210948702746204,
                                -0.1762737567935671, -0.1647720911673137, -0.06558359733649663,
                                0.17421027166502817, -0.19849703567368643, -0.13744422580514634,
                                0.0006715731163110054, 0.01227694470435381, 0.14421781684671128,
                                0.021984452681083764, 0.023775280187172548, 0.018808349035680294,
                                -0.035542234512312074, -0.29509612917900085, -0.06503601372241974,
                                -0.07094652205705643, 0.056632917906556814, -0.06262379033224923, 
                                -0.03285806200334004, -0.004981650272384286, -0.197942459157535, 
                                -0.06309697670595986, 0.04281323615993772, 0.04890420035059963, 
                                -0.020669094752520323, -0.06223668370928083, 0.1935420355626515,
                                -0.021601677739194462, -0.182394517319543, 0.04723444953560829, 
                                0.004966813937893936, 0.2401117788893836, 0.1812194777386529,
                                0.035449533856340816, 0.02681770282132285, -0.18762201070785522,
                                0.11062027301107134, -0.10540414920875005, 0.05349006823131016,
                                0.19285678012030466, 0.07460849519286837, 0.06483855524233409, 
                                -0.03328064722674234, -0.14909936487674713, 0.030711552261241844, 
                                0.0897495730647019, -0.12904933307852065, -0.014361007272132806,
                                0.10396345279046468, -0.07736217762742724, -0.032261730703924386,
                                -0.11976981588772365, 0.23050895120416368, 0.029569712733583792, 
                                -0.11826781396354948, -0.209767113838877, 0.11861431917973927,
                                -0.09038732945919037, -0.11404897059713091, 0.04138824050979955,
                                -0.15966781973838806, -0.13056302389928273, -0.2702060214110783,
                                0.030903535229819163, 0.332784857068743, 0.0830271424991744, 
                                -0.21616241122995103, 0.0646308112357344, -0.07314161317689079,
                                0.010381803116095918, 0.10590656421014241, 0.15453800133296422, 
                                -0.013144881331494876, 0.020249633650694574, -0.09274979255029134,
                                -0.022785415473793234, 0.19045300143105642, -0.0711945721081325, 
                                -0.06348085775971413, 0.19156022582735335, -0.032995903731456826, 
                                0.09997024812868663, 0.02289487501340253, -0.01585422901968871, 
                                -0.04018483829817602, 0.08333102507250649, -0.1010490038565227,
                                -0.0038734166882932186, 0.022165911006075994, -0.06747045634048325,
                                -0.03173697526965823, 0.14158985231603896, -0.12647585357938493, 
                                0.11038230253117425, 0.002957183036154933, 0.0888573985014643,
                                0.013817689448062862, -0.030510292388498783, -0.09006597740309578, 
                                -0.05415193949426923, 0.11655475412096296, -0.1923381324325289, 
                                0.31407363074166433, 0.151382109948567, 0.10588991854872022,
                                0.10671334394386836, 0.1527668216398784, 0.1239716357418469, 
                                -0.006165592265980584, -0.03601385014397757, -0.22865072744233267,
                                -0.019697651267051697, 0.10058920298303876, -0.017874648861054863,
                                0.1051183630313192, 0.004947445155786616]

# dlib预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/Anaconda3/SpyderDataLiu/PicRecgnize/dat/shape_predictor_68_face_landmarks.dat')

# 创建cv2摄像头对象
cap = cv2.VideoCapture(0)

# cap.set(propId, value)
# 设置视频参数，propId设置的视频参数，value设置的参数值
cap.set(3, 480)

# 返回单张图像的128D特征
def get_128d_features(img_gray):
    dets = detector(img_gray, 1)
    if len(dets) != 0:
        shape = predictor(img_gray, dets[0])
        face_descriptor = facerec.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
    return face_descriptor


# cap.isOpened（） 返回true/false 检查初始化是否成功
while cap.isOpened():

    # cap.read()
    # 返回两个值：
    #    一个布尔值true/false，用来判断读取视频是否成功/是否到视频末尾
    #    图像对象，图像的三维矩阵
    flag, im_rd = cap.read()

    # 每帧数据延时1ms，延时为0读取的是静态帧
    kk = cv2.waitKey(1)

    # 取灰度
    img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)

    # 人脸数dets
    dets = detector(img_gray, 0)

    # 待会要写的字体
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(im_rd, "q: quit", (20, 400), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)

    if len(dets) != 0:
        # 检测到人脸

        # 将捕获到的人脸提取特征和内置特征进行比对
        features_rd = get_128d_features(im_rd)
        compare = return_euclidean_distance(features_rd, features_mean_default_person)

        # 让人名跟随在矩形框的下方
        # 确定人名的位置坐标
        pos_text_1 = tuple([dets[0].left(), int(dets[0].bottom()+(dets[0].bottom()-dets[0].top())/4)])

        im_rd = cv2.putText(im_rd, compare.replace("same", "default_person"), pos_text_1, font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

        # 矩形框
        for k, d in enumerate(dets):
            # 绘制矩形框
            im_rd = cv2.rectangle(im_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)

        cv2.putText(im_rd, "faces: " + str(len(dets)), (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    else:
        # 没有检测到人脸
        cv2.putText(im_rd, "no face", (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    # 按下q键退出
    if kk == ord('q'):
        break

    # 窗口显示
    cv2.imshow("camera", im_rd)

# 释放摄像头
cap.release()

# 删除建立的窗口
cv2.destroyAllWindows()
