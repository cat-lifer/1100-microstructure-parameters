# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:20:21 2020

@author: hongyong Han

To: Do or do not. There is no try.

"""
################################ 微观组织参数预测 ################################
##########  f: MLP tanh  alpha=1 (9,) lbfgs
##########  th: KNN  6
##########  V: KNN  3

#导入算法包
from sklearn.neighbors import KNeighborsRegressor  #k-NN
from sklearn.neural_network import MLPRegressor    #神经网络
#导入工具包
import numpy as np
from sklearn.preprocessing import StandardScaler   #预处理 均值为0 方差为1
from sklearn.model_selection import  train_test_split #随机划分
##数据库
f=np.loadtxt(r'C:\Users\Uaena_HY\Desktop\代码集\数据库\f-453.txt',delimiter='\t')
th=np.loadtxt(r'C:\Users\Uaena_HY\Desktop\代码集\数据库\th-453.txt',delimiter='\t')
V=np.loadtxt(r'C:\Users\Uaena_HY\Desktop\代码集\数据库\V-403.txt',delimiter='\t')
#预测集
predict=np.loadtxt(r'C:\Users\Uaena_HY\Desktop\代码集\Testset\组织参数.txt',delimiter='\t')
predictv =predict[:,1:9]
# 预留矩阵
f_pred=np.zeros(shape=(len(predict),50))  #预留矩阵
th_pred=np.zeros(shape=(len(predict),50))  #预留矩阵
V_pred=np.zeros(shape=(len(predict),50))  #预留矩阵

for k in range(50):
    #数据集打乱
    data1=np.random.permutation(f)
    data2=np.random.permutation(th)
    data3=np.random.permutation(V)
    #分输入输出数据   1:f  2：th  3：V  
    x1 = []
    y1 = []
    for line in range(453):
        x1.append(data1[line][:9])
        y1.append(data1[line][-1])
    X1 = np.array(x1)
    y1 = np.array(y1)

    x2 = []
    y2 = []
    for line in range(453):
        x2.append(data2[line][:9])
        y2.append(data2[line][-1])
    X2 = np.array(x2)
    y2 = np.array(y2)

    x3 = []
    y3 = []
    for line in range(403):
        x3.append(data3[line][:8])
        y3.append(data3[line][-1])
    X3 = np.array(x3)
    y3 = np.array(y3)
        
    x_train1,x_test1,y_train1,y_test1 = train_test_split(X1,y1,test_size=0.2,random_state=1)
    x_train2,x_test2,y_train2,y_test2 = train_test_split(X2,y2,test_size=0.2,random_state=1)
    x_train3,x_test3,y_train3,y_test3 = train_test_split(X3,y3,test_size=0.2,random_state=1)        
    ## f  需要归一化
    scaler = StandardScaler()
    scaler.fit(x_train1)
    X_train_scaled1 = scaler.transform(x_train1)
    X_test_scaled1 = scaler.transform(x_test1)
    predict_scaled = scaler.transform(predict)
    ##利用优化的超参数建模
    FeretRatio = MLPRegressor(random_state=38,max_iter=10000,activation ='tanh',
                       alpha=1,hidden_layer_sizes=(9,),solver='lbfgs')
    modelf = FeretRatio.fit(X_train_scaled1,y_train1)
    
    thickness = KNeighborsRegressor(n_neighbors=6)
    modelth = thickness.fit(x_train2,y_train2)
    
    Volume = KNeighborsRegressor(n_neighbors=3)
    modelV = Volume.fit(x_train3,y_train3)

    ##评分
    print("FeretRatio Test R^2: {:.2f}".format(FeretRatio.score(X_test_scaled1, y_test1)))
    print("thickness Test R^2: {:.2f}".format(thickness.score(x_test2, y_test2)))
    print("Volume Test R^2: {:.2f}".format(Volume.score(x_test3, y_test3)))
    print('\n')
    
    ##预测
    f_pred[:,k] = modelf.predict(predict_scaled) 
    th_pred[:,k] = modelth.predict(predict)
    V_pred[:,k] = modelV.predict(predictv)
   
f_results = np.mean(f_pred,axis=1)
th_results = np.mean(th_pred,axis=1)
V_results = np.mean(V_pred,axis=1)

Final_results = np.concatenate((predict,f_results.reshape(len(predict),1),
                                th_results.reshape(len(predict),1),
                                V_results.reshape(len(predict),1)),axis=1)