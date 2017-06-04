# -*- coding: utf-8 -*-
"""
Created on Fri Jun 02 05:47:10 2017

@author: Young
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
=======================================
Lazzy learning
=======================================
Input: training set D
       query point x_q, or validation set
       Kmax - the maximum number of neighbors
Output: y_q - the prediction of the vectorial output of the query point x_q

Steps:
    1. Sort increaingly the set vectors {xi} with respect to the diatance to x_1
    2. [j] will designate the index of the jth closest neighbor of x_q
    3. for k in {2,...,Kmax} do:
        y_qk = sum(y[j])/k
              # where using [j] collects all the k closest neighbor of x_q
        E_look = e_look.mean()
              # where e_look = e_k.mean(), e_k = k*(y[j]-y_qk)/(k-1)
        end
    4. K* = arg min{E_look}
    5. y_q = y_qK*
"""

def elu_distance(x_q, X):
    """
    计算查询点x_q与训练样本X中各店的欧拉距离
    """
    return ((x_q - X)**2).sum(axis = 1)

def abs_distance(x_q, X):
    """
    计算查询点x_q与训练样本X中各店的绝对距离
    """
    return (abs(x_q - X)).sum(axis = 1)

def err_evaluation(y_pred,y,err = 'abs'):
    if err == 'square':
        return ((y_pred-y)**2).mean()
    else:
        return (abs(y_pred-y)).mean()

def lazzy_loo(x_q, X, y, Kmax = 50, dis = elu_distance):
    """
    留一法，默认欧拉距离
    """
    l = len(y)
    if dis == elu_distance:
        distance = elu_distance(x_q, X)
    else:
        distance = abs_distance(x_q, X)
    neighbors = distance.argsort()
    models = []    # 用模型记录误差及相应的邻居数
    for k in xrange(2, Kmax+1):
        e_look = 0.0
        k_neighbors_idx = neighbors[0:k]
        y_qk = y[k_neighbors_idx].mean(axis = 0)
        for j in k_neighbors_idx:           
#            e_loo_kj = 0.0
#            e_loo_kj = k * (y[j] - y_qk) / (k - 1)
#            e_look += sum(e_loo_kj**2)
            # square err
            e_look += sum((k * (y[j] - y_qk) / (k - 1))**2)
#            # absolute err
#            e_look += sum(abs(k * (y[j] - y_qk) / (k - 1)))
        models.append((e_look/k/l, k))
    return models

def lazzy_prediction(x, X, Y, models, method = 'WIN', dis = elu_distance):
    """
    根据学习的models = lazzy_loo(x, X, Y, Kmax)，进行预测
    method = 'M'为多模型平均
    method = 'WM'为加权平均
    method = 'WIN'为选择最佳模型
    """
    
    if dis == elu_distance:
        distance = elu_distance(x, X)
    else:
        distance = abs_distance(x, X)

    neighbors_idx = distance.argsort()
    
    if method == 'WIN':
        models.sort()
        num_neighbors = models[0][1]
        y_pred = Y[neighbors_idx[0:num_neighbors]].mean(axis = 0)
        return y_pred
   
    if method == 'M':
        n = len(models)
        y_pred = 0.0
        for err,num_neighbors in models:
            y_pred += Y[neighbors_idx[0:num_neighbors]].mean(axis =0)
        return y_pred/n
    
    if method == 'WM':
        y_pred = 0.0
        total_err = 0.0
        models.sort()
        err_sorted = sorted([err for err, k in models],reverse = True)
        print err_sorted
        i = 0
        for err,num_neighbors in models:
            y_pred += err_sorted[i] * Y[neighbors_idx[0:num_neighbors]].mean(axis =0)
            total_err += err
            i += 1
        return y_pred/total_err  

def create_dataset(seq, input_lags, pre_period):
    """
    功能：根据时间序列，及给定的输入时滞及预测时长，构建训数据集(X,Y)
    """
    X = []; Y = []
    n = len(seq)
    window = input_lags + pre_period
    for i in xrange(n - window + 1):
#        # if do like this, you need to pay attention
        x = seq[i: input_lags + i]
#or     y = seq[input_lags + i: input_lags + pre_period + i]
        y = seq[input_lags + i: window + i]
        
#        # easy to understand
#        x_y = seq[i:i+window]
#        x = x_y[0:input_lags]
#        y = x_y[input_lags:window]

        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
def choose_best_lag(seq, pre_period, lags = range(1,30), Kmax = 200):
    """
    选择最佳lazzy model,及输入时滞
    """
    models = []
    # 标准化
    std_sca = StandardScaler().fit(np.array(seq).reshape(-1,1))
#    rob_sca = RobustScaler().fit(np.array(seq).reshape(-1,1))
    seq = std_sca.transform(np.array(seq).reshape(-1,1))
    
    # 根据时滞及序列创建数据集,并进行交叉验证
    for input_lag in lags:
#        window = input_lag + pre_period
        X, Y = create_dataset(seq.flatten(), input_lag, pre_period)
        lazzy_models = lazzy_loo(X[-1], X[0:-1], Y[:-1], Kmax)
        y_pred = lazzy_prediction(X[-1], X[0:-1], Y[:-1], lazzy_models)
        err = err_evaluation(y_pred, Y[-1])
        lazzy_models.sort()
        models.append((err, input_lag, lazzy_models[0][1] ))
    models.sort()
    best_lag = models[0][1]
    best_k = models[0][2]
    return models, best_lag, best_k      

if __name__ == '__main__':
    
##    # test elu_distance()
##    a = np.array([1,2,3])
##    b = np.array([[4,5,6],[1,2,3]])
##    print elu_distance(a,b)
#    
##    # 测试lazzy模型
##    def seq2_reg(first=0,second=1,length=1000):
##        """
##        产生二阶自相关时间序列，用于测试lazzy模型
##        """
##        auto = [0]*length;
##        auto[0]=first;auto[1]=second
##        for i in xrange(2,length):
###            auto[i] = 0.6*auto[i-1] + 0.2*auto[i-2] + np.random.randn()
##            auto[i] = 0.6*auto[i-1] + 0.2*auto[i-2] + 0.5*np.random.randn()
##        return auto
#    
#    # 构建训练数据集
#    
#    pre_period = 30
#
##    seq = seq2_reg()
##    np.array(seq).dump('seq2order')
#    seq = np.load('seq2order')
#    seq_test = seq[-pre_period:]
#    seq_train_cv = seq[:-pre_period]
#    
#    lag_models, best_lag, best_k = choose_best_lag(seq_train_cv, pre_period, lags = range(1,50), Kmax = 200)
#    input_lags = best_lag
#    window = input_lags + pre_period
#    
#    from sklearn.preprocessing import StandardScaler
#    from sklearn.preprocessing import RobustScaler
#    std_sca = StandardScaler().fit(np.array(seq_train_cv).reshape(-1,1))
#    rob_sca = RobustScaler().fit(np.array(seq_train_cv).reshape(-1,1))
#
#    seq_train_cv = std_sca.transform(np.array(seq_train_cv).reshape(-1,1)) 
#
##    # testing the function: create_dataset()
##    a = np.array([1,2,3,4,5,6,7])
##    X, Y = create_dataset(a,2,2)
##    print X,Y
#
#    X, Y = create_dataset(seq_train_cv.flatten(), input_lags, pre_period)
#    
#    # test lazzy_loo(..)
#    x_q = X[-1]
#    models = lazzy_loo(x_q, X[:-1], Y[:-1], Kmax = 100)
# 
##    # testing prediction
##    models.sort()
##    x = seq[-window:-window+input_lags]
##    x = std_sca.transform(np.array(x).reshape(-1,1)).flatten()
#    
##    distance = elu_distance(x, X)
##    neighbors = distance.argsort()
##    k_near_neighbors = neighbors[0:models[0][1]]
##    y_pred = Y[k_near_neighbors].mean(axis = 0)
##    
##    y_pred = std_sca.inverse_transform(y_pred.reshape(-1,1))
##    
##    fig, ax = plt.subplots()
##    ax.plot(seq[-pre_period:],label='real')
##    ax.plot(y_pred,label='lazzy - %s neighbors'% models[0][1])
##    
##    k_near_neighbors = neighbors[0:10]
##    y_pred = Y[k_near_neighbors].mean(axis = 0)
##    
##    y_pred = std_sca.inverse_transform(y_pred.reshape(-1,1))
##
##    ax.plot(y_pred,label='%s neighbors'% len(k_near_neighbors))
#    
#    
#    # testing lazzy_prediction()
#    # 新的样本输入
#    x = seq[-window:-window+input_lags]
#    x = std_sca.transform(np.array(x).reshape(-1,1)).flatten()
##    y_pred = lazzy_prediction(x, X, Y, models=models, method = 'WIN')
##    y_pred = std_sca.inverse_transform(y_pred.reshape(-1,1))
#    
#    # drawing
#    fig, ax = plt.subplots()
#    ax.plot(seq_test,label='real')
##    ax.plot(y_pred,label='WIN - %s neighbors'%models[0][1])
#    # 真正预测时充分利用cv的数据，重新训练
#    models = lazzy_loo(x, X, Y, Kmax = 200)
#    methods = ['WIN','M','WM']
#    for method in methods:
#        y_pred = lazzy_prediction(x, X, Y, models=models, method = method)
#        y_pred = std_sca.inverse_transform(y_pred.reshape(-1,1))
#        if method == 'WIN':
#            ax.plot(y_pred,label='%s - %s neighbors'%(method,models[0][1]))
#        else:
#            ax.plot(y_pred,label='%s - %s models'%(method,len(models)))
#    ax.legend()


# df for dataframe, s for series

    df = pd.read_csv('Tianchi_power.csv')
    df['record_date'] = pd.to_datetime(df['record_date'])

# total power consumption
# 先要把record_date格式转换
    s_power_consumption = df.groupby('record_date')['power_consumption'].sum()
    seq = s_power_consumption.values
    
    pre_period = 30
    seq_test = seq[-pre_period:]
    seq_train_cv = seq[:-pre_period]
    
    lag_models, best_lag, best_k = choose_best_lag(seq_train_cv, pre_period, lags = range(1,120), Kmax = 200)
    input_lags = best_lag
    window = input_lags + pre_period

    std_sca = StandardScaler().fit(np.array(seq_train_cv).reshape(-1,1))
    seq_train_cv = std_sca.transform(np.array(seq_train_cv).reshape(-1,1))
    
    X, Y = create_dataset(seq_train_cv.flatten(), input_lags, pre_period)       
    
    # testing lazzy_prediction()
    # 新的样本输入
    x = seq[-window:-window+input_lags]
    x = std_sca.transform(np.array(x).reshape(-1,1)).flatten()
    
    # drawing
    fig, ax = plt.subplots()
    ax.plot(seq_test,label='real')
    # 真正预测时充分利用cv的数据，重新训练
    models = lazzy_loo(x, X, Y, Kmax = 200)
    methods = ['WIN','M','WM']
    for method in methods:
        y_pred = lazzy_prediction(x, X, Y, models=models, method = method)
        y_pred = std_sca.inverse_transform(y_pred.reshape(-1,1))
        if method == 'WIN':
            ax.plot(y_pred,label='%s - %s neighbors'%(method,models[0][1]))
        else:
            ax.plot(y_pred,label='%s - %s models'%(method,len(models)))
    ax.legend()




