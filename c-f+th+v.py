# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 21:14:45 2019

@author: hhy
"""
import math
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor    #Bagging 元估计  (装袋)
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler




##加载数据  f-453 th-453 V-403
data=np.loadtxt(r'C:\Users\Uaena_HY\Desktop\代码集\数据库\V-453.txt',delimiter='\t') #读入txt数据，以tab为分界
#预测数据
data_Predict=np.loadtxt(r'C:\Users\Uaena_HY\Desktop\代码集\Testset\组织参数.txt',delimiter='\t') #读入txt数据，以tab为分界
#data_Predict =data_Predict[:,1:9] #V时用
#预留矩阵
knn_pre=np.zeros(shape=(len(data_Predict),10))
mlp_pre=np.zeros(shape=(len(data_Predict),10))
bag1_pre=np.zeros(shape=(len(data_Predict),10))  #预留矩阵
bag2_pre=np.zeros(shape=(len(data_Predict),10))  #预留矩阵
bagging_pre=np.zeros(shape=(len(data_Predict),10))  #预留矩阵

#定义误差函数
def get_mse(records_real, records_predict):
        """
        均方误差 估计值与真值 偏差
        """
        if len(records_real) == len(records_predict):
            return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
        else:
            return None
def get_rmse(records_real, records_predict):
        """
        均方根误差：是均方误差的算术平方根
        """
        mse = get_mse(records_real, records_predict)
        if mse:
            return math.sqrt(mse)
        else:
            return None
def get_mre(records_real, records_predict):
        """
        平均相对误差
        """
        if len(records_real) == len(records_predict):
            return sum([abs(x - y) for x, y in zip(records_real, records_predict)]/(y_test)) / len(records_real)
        else:
            return None

##建模
for k in range(10):
    #数据集打乱
    data=np.random.permutation(data)
    #分输入输出数据
    x = []
    y = []
    for line in range(453):
        x.append(data[line][:9])
        y.append(data[line][-1])
    X = np.array(x)
    y = np.array(y)
    
    
    #将所有数据分为两类，80%作训练集，20%作预测集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=38)
    #数据预处理 均值为0 方差为1
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 两个模型     #f:tanh   
    knn =  KNeighborsRegressor(n_neighbors=6)
    mlp = MLPRegressor(random_state=38,max_iter=10000,activation ='tanh',alpha=1,hidden_layer_sizes=(9,),solver='lbfgs')
    
    ## Bagging  ##th V bagging只用knn
    bag1 = BaggingRegressor(knn,n_estimators=100,max_samples=0.95,max_features=1.0)
    bag2 = BaggingRegressor(mlp,n_estimators=100,max_samples=0.95,max_features=1.0)
    
    #训练
    knn.fit(X_train, y_train)
    mlp.fit(X_train_scaled, y_train)
    bag1.fit(X_train, y_train)
    bag2.fit(X_train_scaled, y_train)
    
    
    #测试
    knn_test= knn.fit(X_train, y_train).predict(X_test)
    mlp_test = mlp.fit(X_train_scaled, y_train).predict(X_test_scaled)
    combination_test= (knn_test+mlp_test)/2
    bag1_test= bag1.fit(X_train, y_train).predict(X_test)
    bag2_test = bag2.fit(X_train_scaled, y_train).predict(X_test_scaled)
    bagging_test = (bag1_test+bag2_test)/2
    
    ll = len(y_test)
    data_test=np.hstack((y_test.reshape(ll,1),knn_test.reshape(ll,1),mlp_test.reshape(ll,1),
                         bag1_test.reshape(ll,1),bag2_test.reshape(ll,1),
                         bagging_test.reshape(ll,1)))
    
    ##误差  1:knn  2:mlp
    knn_mse = get_mse(y_test, knn_test)
    mlp_mse = get_mse(y_test, mlp_test)
    bag1_mse = get_mse(y_test, bag1_test)
    bag2_mse = get_mse(y_test, bag2_test)
    bag_mse = get_mse(y_test, bagging_test)
    data_mse = np.hstack((knn_mse,mlp_mse,bag1_mse,bag2_mse,bag_mse))
    
    knn_rmse = get_rmse(y_test, knn_test)
    mlp_rmse = get_rmse(y_test, mlp_test)
    bag1_rmse = get_rmse(y_test, bag1_test)
    bag2_rmse = get_rmse(y_test, bag2_test)
    bag_rmse = get_rmse(y_test, bagging_test)
    data_rmse = np.hstack((knn_rmse,mlp_rmse,bag1_rmse,bag2_rmse,bag_rmse))
    
    knn_mre = get_mre(y_test, knn_test)
    mlp_mre = get_mre(y_test, mlp_test)
    bag1_mre = get_mre(y_test, bag1_test)
    bag2_mre = get_mre(y_test, bag2_test)
    bag_mre = get_mre(y_test, bagging_test)
    data_mre = np.hstack((knn_mre,mlp_mre,bag1_mre,bag2_mre,bag_mre))
    
    data_error=np.hstack((data_mre.reshape(5,1),data_mse.reshape(5,1),data_rmse.reshape(5,1)))
                     
    ##评分
    print("knn Train R^2: {:.2f}".format(knn.score(X_train, y_train)))
    print("knn Test R^2: {:.2f}".format(knn.score(X_test, y_test)))
    
    print("mlp Train R^2: {:.2f}".format(mlp.score(X_train_scaled, y_train)))
    print("mlp Test R^2: {:.2f}".format(mlp.score(X_test_scaled, y_test)))
    
    print("bag_knn Train R^2: {:.2f}".format(bag1.score(X_train, y_train)))
    print("bag_knn Test R^2: {:.2f}".format(bag1.score(X_test, y_test)))
    
    print("bag_mlp Train R^2: {:.2f}".format(bag2.score(X_train_scaled, y_train)))
    print("bag_mlp Test R^2: {:.2f}".format(bag2.score(X_test_scaled, y_test)))
    print('\n')
    
    
    ## 预测
    data_Predict_scaled = scaler.transform(data_Predict)
    
    knn_pre[:,k]= knn.fit(X_train, y_train).predict(data_Predict) 
    mlp_pre[:,k]= mlp.fit(X_train_scaled, y_train).predict(data_Predict_scaled)
    bag1_pre[:,k]= bag1.fit(X_train, y_train).predict(data_Predict)    ## th V
    bag2_pre[:,k] = bag2.fit(X_train_scaled, y_train).predict(data_Predict_scaled)
    bagging_pre[:,k] = (bag1_pre[:,k]+bag2_pre[:,k])/2         ##f

##Final reasults
mean_knn_pre=np.mean(knn_pre,1)
mean_mlp_pre=np.mean(mlp_pre,1)
mean_bag1_pre=np.mean(bag1_pre,1) ## th V
mean_bag2_pre=np.mean(bag2_pre,1)
mean_bagging_pre=np.mean(bagging_pre,1) ##f

