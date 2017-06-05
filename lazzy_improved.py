# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 11:21:02 2017

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
    留一法求给定最大邻居数下，所有备选模型：(误差，邻居数)；
    默认欧拉距离
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
#        print err_sorted
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
def choose_best_lag(seq, pre_period, lags = range(1,30), Kmax = 200):
    """
    选择最佳lazzy model,及输入时滞
    模型：(误差，延时，邻居数)
    """
    models = []
    # 标准化
    std_sca = StandardScaler().fit(np.array(seq).reshape(-1,1))
#    rob_sca = RobustScaler().fit(np.array(seq).reshape(-1,1))
    seq = std_sca.transform(np.array(seq).reshape(-1,1))
    
    # 根据时滞及序列创建数据集,并进行交叉验证
    from sklearn.model_selection import train_test_split
    for input_lag in lags:
#        window = input_lag + pre_period
#        X, Y = create_dataset(seq.flatten(), input_lag, pre_period)
#        lazzy_models = lazzy_loo(X[-1], X[0:-1], Y[:-1], Kmax)
#        y_pred = lazzy_prediction(X[-1], X[0:-1], Y[:-1], lazzy_models)
#        err = err_evaluation(y_pred.flatten(), Y[-1])
#
#        lazzy_models.sort()
#        models.append((err, input_lag, lazzy_models[0][1]))
        
#         do more cv
        err = 0.0
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.01, random_state=0)
        for x_q,y_q in zip(X_test,y_test):
            lazzy_models = lazzy_loo(x_q, X_train, y_train, Kmax)
            y_pred = lazzy_prediction(x_q, X_train, y_train, lazzy_models)
            err += err_evaluation(y_pred.flatten(), y_q)
        lazzy_models.sort()
        models.append((err/len(X_test), input_lag, lazzy_models[0][1]))
    models.sort()
    best_lag = models[0][1]
    best_k = models[0][2]
#    fig, ax = plt.subplots()
#    ax.plot(y_pred.flatten(),label='prediction')
#    ax.plot(Y[-1],label='real')
#    ax.set_title('best cv lags')
    return models, best_lag, best_k

def seq_restore(x0, seq):
    n = len(seq)
    y_pred = np.zeros(n)
    y_pred[0] = x0 + seq[0]
    for i in xrange(1,n):
        y_pred[i] = seq[i] + y_pred[i-1]
    return y_pred    

if __name__ == '__main__':
    
# df for dataframe, s for series

    df = pd.read_csv('Tianchi_power.csv')
    df['record_date'] = pd.to_datetime(df['record_date'])

# total power consumption
# 先要把record_date格式转换
    s_power_consumption = df.groupby('record_date')['power_consumption'].sum()
#    seq = s_power_consumption.values
#    power1 = s_power_consumption.values[:-1]
#    power2 = s_power_consumption.values[1:]
#    
#    power = power2 - power1
    power = np.log(s_power_consumption.values)
    
    from statsmodels.tsa.seasonal import seasonal_decompose    
    decomposition = seasonal_decompose(power,freq=7)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
#    print 'Ok'
    trend = np.concatenate((np.tile(trend[3],3),trend[3:-3],np.tile(trend[-4],3))) # 首尾需要合理填充
    residual = power - trend - seasonal

    # for trend
    seq = trend
    
    pre_period = 30
    seq_test = seq[-pre_period:]
    seq_train_cv = seq[:-pre_period]
    
    lag_models, best_trend_lag, best_k = choose_best_lag(seq_train_cv, pre_period,
                                                         lags = range(1,120), Kmax = 200)
    input_lags = best_trend_lag
#    input_lags = 30
    window = input_lags + pre_period
#
    std_sca = StandardScaler().fit(np.array(seq).reshape(-1,1))   # fit all seq
    seq_train_cv = std_sca.transform(np.array(seq_train_cv).reshape(-1,1))
    
    X, Y = create_dataset(seq_train_cv.flatten(), input_lags, pre_period)       
    
    # testing lazzy_prediction()
    # 新的样本输入
    x = seq[-window:-window+input_lags]
##    print 'x1',x
    x = std_sca.transform(np.array(x).reshape(-1,1)).flatten()
##    print 'x2',x
#    
    # drawing
    fig, ax = plt.subplots()
    ax.plot(seq_test,label='real')
    # 真正预测时充分利用cv的数据，重新训练
    models = lazzy_loo(x, X, Y, Kmax = 200)  # 对于趋势Kmax可以作为一个参数调节
#    print 'x3',x
    methods = ['WIN','M','WM']
#    methods = ['WM','M','WIN']
    for method in methods:
        y_pred = lazzy_prediction(x, X, Y, models=models, method = method)
        y_pred = std_sca.inverse_transform(y_pred.reshape(-1,1))
        if method == 'WIN':
            err = (abs(y_pred-seq_test)/seq_test).mean()
            ax.plot(y_pred,label='%s - %s neighbors with err %.2f%%'%(method,models[0][1],100*err))
        else:
            err = (abs(y_pred-seq_test)/seq_test).mean()
            ax.plot(y_pred,label='%s - %s models with err %.2f%%'%(method,len(models),100*err))
    ax.legend()
    ax.set_title('trend')
    
    y_trend = y_pred

#    from sklearn.ensemble import RandomForestRegressor
#    X, Y = create_dataset(seq_train_cv.flatten(), input_lags, pre_period)     
#    reg = RandomForestRegressor(verbose=True,max_features = 'auto',min_samples_split=2)
#    reg.fit(X,Y)
#    # 新的样本输入
#    x = seq[-window:-window+input_lags]
#    x = std_sca.transform(np.array(x).reshape(-1,1)).flatten()
#    y_pred = reg.predict(x)
#    y_pred = std_sca.inverse_transform(y_pred.reshape(-1,1))    
#    fig, ax = plt.subplots()
#    ax.plot(seq_test,label='real')
#    ax.plot(y_pred,label='pridition')
#    ax.legend()
#    ax.set_title('DT')
#    
#    err = 100*(abs(y_pred.flatten()-seq_test)/seq_test).mean()
#    print 'trend testing err: %.2f%%'% err
#    
#    from sklearn.neural_network import MLPRegressor
#    
#    reg = MLPRegressor(activation = 'relu',hidden_layer_sizes = (10,),
#                       max_iter=10000,verbose=False,learning_rate='adaptive',
#                       tol=0.0,warm_start=True,solver='adam')
#    
#    reg.fit(X,Y)
#    y_pred = reg.predict(x)
#    y_pred = std_sca.inverse_transform(y_pred.reshape(-1,1))    
#    fig, ax = plt.subplots()
#    ax.plot(seq_test,label='real')
#    ax.plot(y_pred,label='pridition')
#    ax.legend()
#    ax.set_title('ANN')
    
    err = 100*(abs(y_pred.flatten()-seq_test)/seq_test).mean()
    print 'trend testing err: %.2f%%'% err
    
    # for residual
    seq = residual
    
    pre_period = 30
    seq_test = seq[-pre_period:]
    seq_train_cv = seq[:-pre_period]
    
    lag_models, best_resi_lag, best_k = choose_best_lag(seq_train_cv, pre_period, lags = range(1,120), Kmax = 200)
    input_lags = best_resi_lag
#    input_lags = 86
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
    models = lazzy_loo(x, X, Y, Kmax = 100)
    
    methods = ['WIN','WM','M']
    methods = ['M','WM','WIN']
    for method in methods:
        y_pred = lazzy_prediction(x, X, Y, models=models, method = method)
        y_pred = std_sca.inverse_transform(y_pred.reshape(-1,1))
        if method == 'WIN':
            err = (abs(y_pred-seq_test)).mean()
            ax.plot(y_pred,label='%s - %s neighbors with err %.2f'%(method,models[0][1],err))
        else:
            err = (abs(y_pred-seq_test)).mean()
            ax.plot(y_pred,label='%s - %s models with err %.2f'%(method,len(models),err))
    ax.legend()
    ax.set_title('residual')
    
    y_resi = y_pred
    
    err = (abs(y_pred-seq_test)).mean()
    print 'residual testing err: %.2f'% err
    
    # restore the predictions
    y_pred = y_resi + y_trend + seasonal[-pre_period:].reshape(-1,1)
    # drawing
    fig, ax = plt.subplots()
    ax.plot(power[-pre_period:],label='real')
    ax.plot(y_pred,label='prediction')
    ax.legend()
    
    err = (abs(y_pred-power[-pre_period:])/power[-pre_period:]).mean()
    print 'testing err: %s'% err
    print 'WM + M'
    
#    y = seq_restore(power2[-30],y_pred.flatten())
#    # drawing
#    fig, ax = plt.subplots()
#    ax.plot(power2[-pre_period+1:],label='real')
#    ax.plot(y_pred,label='prediction')
#    ax.legend()

    """
    ======================================================
    Prediction
    ======================================================
    """
    # for trend
    seq = trend
    
    pre_period = 30
    
    input_lags = best_trend_lag
#    input_lags = 3
    window = input_lags + pre_period

    std_sca = StandardScaler().fit(np.array(seq).reshape(-1,1))
    seq = std_sca.transform(np.array(seq).reshape(-1,1))
    
    X, Y = create_dataset(seq.flatten(), input_lags, pre_period)       
    
    # testing lazzy_prediction()
    # 新的样本输入
    x = seq[-window:-window+input_lags]  # or X[-1]
    x = std_sca.transform(np.array(x).reshape(-1,1)).flatten()
    
    # drawing
    fig, ax = plt.subplots()
#    ax.plot(seq_test,label='real')
    # 真正预测时充分利用cv的数据，重新训练
    models = lazzy_loo(x, X, Y, Kmax = 200)
    methods = ['WIN','M','WM']
#    methods = ['M','WM','WIN']
    for method in methods:
        y_pred = lazzy_prediction(x, X, Y, models=models, method = method)
        y_pred = std_sca.inverse_transform(y_pred.reshape(-1,1))
        if method == 'WIN':
            ax.plot(y_pred,label='%s - %s neighbors'%(method,models[0][1]))
        else:
            ax.plot(y_pred,label='%s - %s models'%(method,len(models)))
    ax.legend()
    ax.set_title('Trend Prediction')
    
    y_trend = y_pred
    
    # for residual
    seq = residual
    
    input_lags = best_resi_lag
#    input_lags = 86
    window = input_lags + pre_period

    std_sca = StandardScaler().fit(np.array(seq).reshape(-1,1))
    seq = std_sca.transform(np.array(seq).reshape(-1,1))
    
    X, Y = create_dataset(seq.flatten(), input_lags, pre_period)       
    
    # testing lazzy_prediction()
    # 新的样本输入
    x = seq[-window:-window+input_lags] # or X[-1]
    x = std_sca.transform(np.array(x).reshape(-1,1)).flatten()
    
    # drawing
    fig, ax = plt.subplots()
#    ax.plot(seq_test,label='real')
    # 真正预测时充分利用cv的数据，重新训练
    models = lazzy_loo(x, X, Y, Kmax = 200)
    methods = ['M','WM','WIN']
    for method in methods:
        y_pred = lazzy_prediction(x, X, Y, models=models, method = method)
        y_pred = std_sca.inverse_transform(y_pred.reshape(-1,1))
        if method == 'WIN':
            ax.plot(y_pred,label='%s - %s neighbors'%(method,models[0][1]))
        else:
            ax.plot(y_pred,label='%s - %s models'%(method,len(models)))
    ax.legend()
    ax.set_title('Residual Prediction')
    
    y_resi = y_pred
    
    # restoring, final prediction
    y_pred = y_resi + y_trend + seasonal[0:30].reshape(-1,1)
     # drawing
    fig, ax = plt.subplots()
    ax.plot(y_pred,label='prediction')
    ax.legend()
    ax.set_title('Final Prediction')
    
##    power9 = y_pred
###
###    # write to file
###    fr = open('Tianchi_power_predict_table.csv','w')
###    fr.write('record_date,power_consumption\n')
###    for i,power in enumerate(power9):
###        if i+1 < 10:
###            fr.write('2016090%s,'%(i+1)+str(int(power))+'\n')
###        else:
###            fr.write('201609%s,'%(i+1)+str(int(power))+'\n')
###    fr.close()
###    
###    plt.plot(power9)
##    



