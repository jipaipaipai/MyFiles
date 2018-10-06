# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 17:54:39 2018

@author: haoyu
"""

'''混淆矩阵，精准率，召回率'''
import numpy as np
from sklearn import datasets
digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()

#使数据集产生极度偏斜
y[digits.target==9] = 1
y[digits.target!=9] = 0
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)
y_log_predict = log_reg.predict(X_test)

def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))
TN(y_test, y_log_predict)

def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))
FP(y_test, y_log_predict)

def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))
FN(y_test, y_log_predict)

def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))
TP(y_test, y_log_predict)

def confusion_matrix(y_true, y_predict):
    return np.array([
                [TN(y_true, y_predict), FP(y_true, y_predict)],
                [FN(y_true, y_predict), TP(y_true, y_predict)]
            ])

confusion_matrix(y_test, y_log_predict)

def precision_score(y_true, y_predict):#精准率
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0.0
precision_score(y_test, y_log_predict)

def recall_score(y_true, y_predict):#召回率
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0
recall_score(y_test, y_log_predict)


##scikit-learn中的混淆矩阵，精准率和召回率
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_log_predict)

from sklearn.metrics import precision_score
precision_score(y_test, y_log_predict)

from sklearn.metrics import recall_score
recall_score(y_test, y_log_predict)

#有时精准率高而召回率低，有时相反，并不会同时高，关注哪一个视情况而定。
#比如股票预测注重精准率，而病人诊断注重召回率


'''F1 Score'''
#若想兼顾精准率和召回率，则用：F1 Score
#F1 Score是precision和recall的调和平均值(1/F1 = 1/2(1/precision + 1/recall))
import numpy as np

def f1_score(precision, recall):
    try:
        return 2 * precision * recall / (precision + recall)
    except:
        return 0.0

precision = 0.5
recall = 0.5
f1_score(precision, recall)

precision = 0.1
recall = 0.9
np.mean([precision, recall])
f1_score(precision, recall)#更好的同时表示两个指标

from sklearn import datasets
digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()

y[digits.target==9] = 1
y[digits.target!=9] = 0

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)
y_predict = log_reg.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_predict)

from sklearn.metrics import precision_score
precision_score(y_test, y_predict)

from sklearn.metrics import recall_score
recall_score(y_test, y_predict)

from sklearn.metrics import f1_score
f1_score(y_test, y_predict)


'''精准率和召回率的平衡，调整阈值来展现'''
import numpy as np
from sklearn import datasets
digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()

y[digits.target==9] = 1
y[digits.target!=9] = 0

from sklearn import datasets
digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()

y[digits.target==9] = 1
y[digits.target!=9] = 0

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_predict = log_reg.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_predict)

from sklearn.metrics import precision_score
precision_score(y_test, y_predict)

from sklearn.metrics import recall_score
recall_score(y_test, y_predict)

from sklearn.metrics import f1_score
f1_score(y_test, y_predict)

#调整阈值，精准率和召回率的变化情况，logistic默认阈值为0
decision_scores = log_reg.decision_function(X_test)
np.min(decision_scores)
np.max(decision_scores)

y_predict_2 = np.array(decision_scores >= 5, dtype='int')
confusion_matrix(y_test, y_predict_2)
precision_score(y_test, y_predict_2)
recall_score(y_test, y_predict_2)

y_predict_3 = np.array(decision_scores >= -5, dtype='int')
confusion_matrix(y_test, y_predict_3)
precision_score(y_test, y_predict_3)
recall_score(y_test, y_predict_3)


'''precision-recall曲线，精准-召回曲线'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()

y[digits.target==9] = 1
y[digits.target!=9] = 0

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
decision_scores = log_reg.decision_function(X_test)#相当于返回与阈值对比的值

#设置阈值thresholds并分别绘制不同阈值情况下的精准曲线和召回曲线
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
precisions = []
recalls = []
thresholds = np.arange(np.min(decision_scores), np.max(decision_scores), 0.1)
for threshold in thresholds:
    y_predict = np.array(decision_scores >= threshold, dtype='int')
    precisions.append(precision_score(y_test, y_predict))
    recalls.append(recall_score(y_test, y_predict))
plt.plot(thresholds, precisions)
plt.plot(thresholds, recalls)
plt.show()

#precision-recall曲线
plt.plot(precisions,recalls)
plt.show()

#scikit-learn中的精准-召回曲线
from sklearn.metrics import precision_recall_curve
#返回3个向量，精确率，召回率和阈值,阈值向量的数比其他两个向量少一个数
precisions, recalls, thresholds = precision_recall_curve(y_test, decision_scores)
plt.plot(thresholds, precisions[:-1])
plt.plot(thresholds, recalls[:-1])
plt.plot(precisions,recalls)
plt.show()


'''ROC曲线:描述TPR和FPR之间的关系
   TPR(True Positive Rate)=TP/(TP+FN)=recall
   FPR(False Positive Rate)=FP/(TN+FP)'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()

y[digits.target==9] = 1
y[digits.target!=9] = 0

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
decision_scores = log_reg.decision_function(X_test)

from MLself.metrics import FPR,TPR
fprs = []
tprs = []
thresholds = np.arange(np.min(decision_scores), np.max(decision_scores), 0.1)
for threshold in thresholds:
    y_predict = np.array(decision_scores >= threshold, dtype='int')
    fprs.append(FPR(y_test, y_predict))
    tprs.append(TPR(y_test, y_predict))

plt.plot(fprs, tprs)
plt.show()

#scikit-learn中的ROC
from sklearn.metrics import roc_curve
fprs, tprs, thresholds = roc_curve(y_test, decision_scores)
plt.plot(fprs, tprs)
plt.show()#ROC与x轴围成的面积越大，ROC表现越好，模型预测越好

from sklearn.metrics import roc_auc_score#求面积
roc_auc_score(y_test, decision_scores)


'''多分类问题中的混淆矩阵'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
digits = datasets.load_digits()
X = digits.data
y = digits.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)

y_predict = log_reg.predict(X_test)
from sklearn.metrics import precision_score
precision_score(y_test, y_predict, average='micro')#调整参数对多分类进行处理
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_predict)
cfm = confusion_matrix(y_test, y_predict)
plt.matshow(cfm, cmap=plt.cm.gray)
plt.show()













