# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 14:44:06 2021

@author: hongyong Han

To: Do or do not. There is no try.

"""
"""
################################ 微观组织参数预测 ################################
 V: model = MLPRegressor(random_state=0,max_iter=5000,activation ='tanh',
                       alpha=0.01,hidden_layer_sizes=(10,),solver='lbfgs')

 th: model = MLPRegressor(random_state=0,max_iter=5000,activation ='relu',
                       alpha=1,hidden_layer_sizes=(10,),solver='lbfgs')

  f: model = MLPRegressor(random_state=0,max_iter=50000,activation ='tanh',
                       alpha=1,hidden_layer_sizes=(8,),solver='lbfgs')
"""

from sklearn.neural_network import MLPRegressor    #神经网络
import numpy as np
from sklearn.preprocessing import StandardScaler   #预处理 均值为0 方差为1
from sklearn.model_selection import  train_test_split #随机划分

##数据库
f=np.loadtxt(r'C:\Users\Uaena_HY\Desktop\代码集\数据库\f-279.txt',delimiter='\t')
th=np.loadtxt(r'C:\Users\Uaena_HY\Desktop\代码集\数据库\th-279.txt',delimiter='\t')
V=np.loadtxt(r'C:\Users\Uaena_HY\Desktop\代码集\数据库\V-279.txt',delimiter='\t')

#预测集
predict=np.loadtxt(r'C:\Users\Uaena_HY\Desktop\代码集\Testset\组织参数.txt',delimiter='\t')
predictv =predict[:,1:9]

# 预留矩阵
f_pred=np.zeros(shape=(len(predict),10))  #预留矩阵
th_pred=np.zeros(shape=(len(predict),10))  #预留矩阵
V_pred=np.zeros(shape=(len(predict),10))  #预留矩阵

for k in range(10):
    #数据集打乱
    data1=np.random.permutation(f)
    data2=np.random.permutation(th)
    data3=np.random.permutation(V)
    
    X1 = data1[:,:9]
    y1 = data1[:,-1]
    
    X2 = data2[:,:9]
    y2 = data2[:,-1]
    
    X3 = data3[:,:8]
    y3 = data3[:,-1]
    
    x_train1,x_test1,y_train1,y_test1 = train_test_split(X1,y1,test_size=0.2,random_state=1)
    x_train2,x_test2,y_train2,y_test2 = train_test_split(X2,y2,test_size=0.2,random_state=1)
    x_train3,x_test3,y_train3,y_test3 = train_test_split(X3,y3,test_size=0.2,random_state=1)
    
    ##利用优化的超参数建模
    FeretRatio = MLPRegressor(random_state=0,max_iter=50000,activation ='tanh',
                       alpha=1,hidden_layer_sizes=(8,),solver='lbfgs')
    modelf = FeretRatio.fit(x_train1,y_train1)
    
    thickness = MLPRegressor(random_state=0,max_iter=5000,activation ='relu',
                       alpha=1,hidden_layer_sizes=(10,),solver='lbfgs')
    modelth = thickness.fit(x_train2,y_train2)
    
    Volume = MLPRegressor(random_state=0,max_iter=5000,activation ='tanh',
                       alpha=0.01,hidden_layer_sizes=(10,),solver='lbfgs')
    modelV = Volume.fit(x_train3,y_train3)
    
    ##评分
    print("FeretRatio Test R^2: {:.2f}".format(FeretRatio.score(x_test1, y_test1)))
    print("thickness Test R^2: {:.2f}".format(thickness.score(x_test2, y_test2)))
    print("Volume Test R^2: {:.2f}".format(Volume.score(x_test3, y_test3)))
    print('\n')
    
    ##预测
    f_pred[:,k] = modelf.predict(predict) 
    th_pred[:,k] = modelth.predict(predict)
    V_pred[:,k] = modelV.predict(predictv)
   
f_results = np.mean(f_pred,axis=1)
th_results = np.mean(th_pred,axis=1)
V_results = np.mean(V_pred,axis=1)

Final_results = np.concatenate((predict,f_results.reshape(len(predict),1),
                                th_results.reshape(len(predict),1),
                                V_results.reshape(len(predict),1)),axis=1)