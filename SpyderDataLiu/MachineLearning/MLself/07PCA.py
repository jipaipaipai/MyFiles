# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 08:44:10 2018

@author: haoyu
"""

'''使用梯度上升法求解主成分'''
import numpy as np
import matplotlib.pyplot as plt

X = np.empty((100,2))
X[:,0] = np.random.uniform(0., 100, size=100)
X[:,1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 10., size=100)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

#demean
def demean(X):
    return X - np.mean(X, axis=0)

X_demean = demean(X)
plt.scatter(X_demean[:, 0], X_demean[:, 1])
plt.show()

#梯度上升法
def f(w, X):
    return np.sum((X.dot(w)**2)) / len(X)

def df_math(w, X):
    return X.T.dot(X.dot(w)) * 2. / len(X)

def direction(w):
    return w / np.linalg.norm(w)

def gradient_ascent(df, X, initial_w, eta, n_iters = 1e4, epsilon=1e-8):
    
    w = direction(initial_w)
    cur_iter = 0
    
    while cur_iter < n_iters:
        gradient = df(w, X)
        last_w = w
        w = w+eta * gradient
        w = direction(w)#每次求一个单位方向
        if(abs(f(w, X) - f(last_w, X)) < epsilon):
            break
        
        cur_iter += 1
    return w

initial_w = np.random.random(X.shape[1])#不能用0向量开始
eta = 0.001
#不能用standardScaler标准化
w = gradient_ascent(df_math, X_demean, initial_w, eta)

plt.scatter(X_demean[:, 0], X_demean[:, 1])
plt.plot([0, w[0]*30], [0, w[1]*30], color='r')
plt.show()


###获得前n个主成分
X = np.empty((100,2))
X[:,0] = np.random.uniform(0., 100, size=100)
X[:,1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 10., size=100)

def demean(X):
    return X - np.mean(X, axis=0)
X = demean(X)

def f(w, X):
    return np.sum((X.dot(w)**2)) / len(X)

def df(w, X):
    return X.T.dot(X.dot(w)) * 2. / len(X)

def direction(w):
    return w / np.linalg.norm(w)

def first_component(X, initial_w, eta, n_iters = 1e4, epsilon=1e-8):
    
    w = direction(initial_w)
    cur_iter = 0
    
    while cur_iter < n_iters:
        gradient = df(w, X)
        last_w = w
        w = w+eta * gradient
        w = direction(w)#每次求一个单位方向
        if(abs(f(w, X) - f(last_w, X)) < epsilon):
            break
        
        cur_iter += 1
    return w

initial_w = np.random.random(X.shape[1])#不能用0向量开始
eta = 0.01

w = first_component(X, initial_w, eta)

X2 = np.empty(X.shape)
X2 = X - X.dot(w).reshape(-1, 1) * w
#for i in range(len(X)):
#    X2[i] = X[i] - X[i].dot(w) * w

plt.scatter(X2[:, 0], X2[:, 1])
plt.show()

w2 = first_component(X2, initial_w, eta)
w.dot(w2)


def first_n_component(n, X,eta=0.01, n_iters = 1e4, epsilon=1e-8):
    
    X_pca = X.copy()
    X_pca = demean(X_pca)
    res = []
    for i in range(n):
        initial_w = np.random.random(X_pca.shape[1])
        w = first_component(X_pca, initial_w, eta)
        res.append(w)
        
        X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w
        
    return res

first_n_component(2, X)


#封装的类
X = np.empty((100,2))
X[:,0] = np.random.uniform(0., 100, size=100)
X[:,1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 10., size=100)

from MLself.PCA import PCA
pca = PCA(n_components=2)
pca.fit(X)
pca.components_

pca = PCA(n_components=1)
pca.fit(X)
X_reduction = pca.transform(X)
X_reduction.shape
X_restore = pca.inverse_transform(X_reduction)
X_restore.shape


###scikit-learn中的PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(X)
pca.components_
X_reduction = pca.transform(X)

###实例
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

from sklearn.neighbors import KNeighborsClassifier
'''%%time'''
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
knn_clf.score(X_test, y_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
'''%%time'''
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction, y_train)#时间明显变快
knn_clf.score(X_test_reduction, y_test)#精度明显变低

pca.explained_variance_ratio_#方差贡献率

###
pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)
pca.explained_variance_ratio_

plt.plot([i for i in range(X_train.shape[1])],
         [np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(X_train.shape[1])])
plt.show()

pca = PCA(0.95)
pca.fit(X_train)
pca.n_components_
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
'''%%time'''
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction, y_train)
knn_clf.score(X_test_reduction, y_test)

#当主成分数为2时
for i in range(10):
    plt.scatter(X_train_reduction[y_train==i,0], X_train_reduction[y_train==i, 1], alpha=0.8)
plt.show()


'''MNIST数据集'''
import numpy as np
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home=r'D:\Anaconda3\SpyderDataLiu\Python机器学习')
X, y = mnist['data'], mnist['target']
X.shape
X_train = np.array(X[:60000], dtype=float)
y_train = np.array(y[:60000], dtype=float)
X_test = np.array(X[60000:], dtype=float)
y_test = np.array(y[60000:], dtype=float)

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
knn_clf.score(X_test,y_test)

from sklearn.decomposition import PCA
pca = PCA(0.9)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
knn_clf = KNeighborsClassifier
knn_clf.fit(X_train_reduction, y_train)
knn_clf.score(X_test_reduction, y_test)#时间快了非常多


'''使用PCA降噪'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
digits = datasets.load_digits()
X = digits.data
y = digits.target
noisy_digits = X + np.random.normal(0, 4, size=X.shape)
example_digits = noisy_digits[y==0,:][:10]

for num in range(1,10):
    X_num = noisy_digits[y==num,:][:10]
    example_digits = np.vstack([example_digits, X_num])
example_digits.shape

def plot_digits(data):
    fig, axes = plt.subplots(10, 10, figsize=(10, 10),
                             subplot_kw={'xticks':[],'yticks':[]},
                             gridspec_kw=dict(hspace=0.1,wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap='binary',interpolation='nearest',
                  clim=(0, 16))
    plt.show()

plot_digits(example_digits)


pca = PCA(0.5)
pca.fit(noisy_digits)
pca.n_components_
components = pca.transform(example_digits)
filtered_digits = pca.inverse_transform(components)
plot_digits(filtered_digits)


'''特征脸'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(r'D:\Anaconda3\SpyderDataLiu\Python机器学习')
random_indexes = np.random.permutation(len(faces.data))
X = faces.data[random_indexes]
example_faces = X[:36,:]

def plot_faces(faces):
    fig, axes = plt.subplots(6, 6, figsize=(10, 10),
                             subplot_kw={'xticks':[],'yticks':[]},
                             gridspec_kw=dict(hspace=0.1,wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(faces[i].reshape(62, 47),
                  cmap='bone')
    plt.show()

plot_faces(example_faces)
faces.target_names

from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized')
pca.fit(X)
pca.components_
plot_faces(pca.components_[:36,:])


faces2 = fetch_lfw_people(min_faces_per_person=60,
                          data_home=r'D:\Anaconda3\SpyderDataLiu\Python机器学习')










