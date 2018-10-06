# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:12:07 2018

@author: haoyu
"""

'''集成学习,ensemble learning'''

'''hard voting:少数服从多数'''
#手动投票
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)
log_clf.score(X_test, y_test)

from sklearn.svm import SVC
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_clf.score(X_test, y_test)

from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
dt_clf.score(X_test, y_test)

y_predict1 = log_clf.predict(X_test)
y_predict2 = svm_clf.predict(X_test)
y_predict3 = dt_clf.predict(X_test)
y_predict = np.array((y_predict1 + y_predict2 + y_predict3) >= 2, dtype='int')
y_predict[:10]

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict)

#scikit-learn中的VotingClassifier
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators=[
        ('log_clf', LogisticRegression()),
        ('svm_clf', SVC()),
        ('dt_clf', DecisionTreeClassifier())
        ], voting='hard')#hard:少数服从多数的投票
voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)


'''soft voting:有权值的投票'''
'''要求集合的每一个模型都能估计概率'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

#hard voting
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators=[
        ('log_clf', LogisticRegression()),
        ('svm_clf', SVC()),
        ('dt_clf', DecisionTreeClassifier(random_state=666))
        ], voting='hard')#hard:少数服从多数的投票
voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)

#soft voting
voting_clf2 = VotingClassifier(estimators=[
        ('log_clf', LogisticRegression()),
        ('svm_clf', SVC(probability=True)),#SVC默认不支持计算概率
        ('dt_clf', DecisionTreeClassifier(random_state=666))
        ], voting='soft')#hard:少数服从多数的投票
voting_clf2.fit(X_train, y_train)
voting_clf2.score(X_test, y_test)


'''Bagging(放回取样)和Pasting(不放回取样),Bagging更常用
   Bagging可以生成成百上千个子模型，Pasting只能生成有限个
   统计学中Bagging称为bootstrap:使用同一种分类器，但是产生多个子模型'''
#在scikit-learn中使用Bagging
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
#使用Baggin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
bagging_clf = BaggingClassifier(DecisionTreeClassifier(),
                                n_estimators=500, #生成的子模型的个数
                                max_samples=100,#每个子模型包含的样本数量
                                bootstrap=True)#放回取样
bagging_clf.fit(X_train, y_train)
bagging_clf.score(X_test, y_test)


'''OOB : Out-of-Bag'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
#使用oob
#此处将使用决策树作为base estimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
bagging_clf = BaggingClassifier(DecisionTreeClassifier(),
                                n_estimators=500, #生成的子模型的个数
                                max_samples=100,#每个子模型包含的样本数量
                                bootstrap=True,#放回取样
                                oob_score=True)#将记录从未被取样的样本，作为之后的测试集
bagging_clf.fit(X, y)
bagging_clf.oob_score_

#bootstrap_features:对特征进行随机采样，可用于多特征数据，如图像识别
#只对特征随机采样
random_subspaces_clf = BaggingClassifier(DecisionTreeClassifier(),
                                n_estimators=500, #生成的子模型的个数
                                max_samples=500,#每个子模型包含的样本数量
                                bootstrap=True,#放回取样
                                oob_score=True,#将记录从未被取样的样本，作为之后的测试集
                                max_features=1,#每次取样的特征数
                                bootstrap_features=True)#对每个样本的特征进行随机取样
random_subspaces_clf.fit(X, y)
random_subspaces_clf.oob_score_

#对样本和特征都进行随机采样
random_subspaces_clf = BaggingClassifier(DecisionTreeClassifier(),
                                n_estimators=500, #生成的子模型的个数
                                max_samples=100,#每个子模型包含的样本数量
                                bootstrap=True,#放回取样
                                oob_score=True,#将记录从未被取样的样本，作为之后的测试集
                                max_features=1,#每次取样的特征数
                                bootstrap_features=True)#对每个样本的特征进行随机取样
random_subspaces_clf.fit(X, y)
random_subspaces_clf.oob_score_
###以上将决策树作为base estimator的bagging方法称为随机森林


#scikit-learn已经封装了随机森林
'''随机森林和Extra-Trees
   Extra-Trees:极其随机树：每一个子模型（决策树）在节点划分上，
   使用随机的特征和随机的阈值，使得每一棵树的形状差异都更加大'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=666)
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
#随机森林
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=500, random_state=666,
                                oob_score=True, n_jobs=-1)
rf_clf.fit(X, y)
rf_clf.oob_score_

#Extra-Trees:抑制了过拟合，但是增大了bias
from sklearn.ensemble import ExtraTreesClassifier
et_clf = ExtraTreesClassifier(n_estimators=500, bootstrap=True, oob_score=True,
                              random_state=666)
et_clf.fit(X, y)
et_clf.oob_score_


'''集成学习解决回归问题'''
#from sklearn.ensemble import BaggingRegressor
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import ExtraTreesRegressor


'''AdaBoosting:逐步完善模型，将上一次模型预测差别大的点赋大的权值，
   对全部数据进行再次拟合。
   也需要base estimator'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=666)
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier#作为base estimator
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=500)
ada_clf.fit(X_train, y_train)
ada_clf.score(X_test, y_test)


'''Gradient Boosting:规定将决策树作为base estimator
   同样是逐步完善模型，但是与AdaBoosting不同的是每次完善的拟合只对误差项进行，
   最后将各个模型加起来得到最后的模型'''
from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(max_depth=2, n_estimators=30)
gb_clf.fit(X_train, y_train)
gb_clf.score(X_test, y_test)














