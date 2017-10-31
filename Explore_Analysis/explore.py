# -*- coding: utf-8 -*-
"""
Created on Mon May 29 08:09:45 2017

@author: Young
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression 
from numpy.linalg import pinv

# df for dataframe, s for series
df = pd.read_csv('Tianchi_power.csv')
df['record_date'] = pd.to_datetime(df['record_date'])

# total power consumption
s_power_consumption = df.groupby('record_date')['power_consumption'].sum()
#s_power_consumption.index = pd.to_datetime(s_power_consumption.index).sort_values()

def corr_xy_plot(s, k):
    """
    观测时间序列自相关性
    """
    n = len(s)
    for i in range(0,n-k):
        plt.scatter(s[i],s[i+k])
    plt.title("lag=%s"%k)
    plt.show()
#corr_xy_plot(s_power_consumption.values,1)
#corr_xy_plot(s_power_consumption.values,3)
#corr_xy_plot(s_power_consumption.values,5)
#corr_xy_plot(s_power_consumption.values,7)
#corr_xy_plot(s_power_consumption.values,9)

def corr_fit_plot(s, k):
    """
    求时间序列相关系数lag=k；
    可视化
    """
    n = len(s)
    x = []; y = []
    for i in range(0,n-k):
        x.append([s[i]])
        y.append([s[i+k]])
    plt.scatter(x,y)
    
#    # using sklearn
#    re = LinearRegression()
#    re.fit(x,y)
#    pred = re.predict(x)
#    coef = re.coef_
#    plt.plot(x,pred,'r-')
    
    # least square by myself
    x = np.array(x)
    y = np.array(y)
    one = np.ones((x.shape[0],1))
    x = np.concatenate((one,np.array(x)),axis=1)
    coefs = np.dot(pinv(x),y)
    pred = coefs[0]*x[:,0] + coefs[1]*x[:,1]
    coef = coefs[1]
    plt.plot(x[:,1],pred,'r-')
    
    plt.title('Corr=%s'%coef+' Lag=%s'%k)
    plt.show()    

    return coef

#corr_fit_plot(s_power_consumption.values,1)
#corr_fit_plot(s_power_consumption.values,3)
#corr_fit_plot(s_power_consumption.values,5)
#corr_fit_plot(s_power_consumption.values,7)
corr_fit_plot(s_power_consumption.values,60)

# 观测时间序列某t时刻的概率分布形状（假设平稳条件下） 密度估计？？
plt.hist(s_power_consumption.values,bins=50)
plt.show()

# 自相关性
def auto_corr(x, l):
    return [x.autocorr(i) for i in l]

corr = auto_corr(s_power_consumption,range(1,60))
plt.stem(corr)
plt.title('Auto Correlation')
plt.show()

# 偏自相关函数
def partial_corr(x, k):
    """
    第k阶偏自相关
    w0 + w(k)*x(1) +w(k-1)*x(2) + ... + w(1)*x(k) = x(k+1)
    """
    n = len(x)
    X = [];Y=[]
    for i in range(0,n-k-1):
        X.append(x[i:i+k])
        Y.append(x[i+k+1])
    X = np.array(X); Y = np.array(Y)
    one = np.ones((X.shape[0],1))
    X = np.concatenate((one,X), axis=1)
    coef = np.dot(pinv(X),Y)
#    print 'coef=%s'%coef
    return coef[1]  # 注意取参数的位置w(k),距离x(k+1)相距k的项：w(k)*x(1)
       
def partial_corrs(x, lag=10):
    coefs = []
    for i in range(1,lag+1):
        coefs.append(partial_corr(x, i))
    return coefs

partial_coefs = partial_corrs(s_power_consumption.values, lag=200)
plt.stem(partial_coefs)
plt.title('Partial Auto Correlation')
plt.show()

## 测试偏自相关函数
#def seq2_regression(first=0,second=1,length=50):
#    auto = [0]*length;
#    auto[0]=first;auto[1]=second
#    for i in xrange(2,length):
#        auto[i] = 0.6*auto[i-1] + 0.2*auto[i-2] + np.random.randn()
#    return auto
#
#x = seq2_regression(length = 200)
#
#corr = auto_corr(pd.Series(data=x),range(1,60))
#plt.stem(corr)
#plt.title('Auto Correlation')
#plt.show()
#
#partial_coefs = partial_corrs(x, lag=30)
#plt.stem(partial_coefs)
#plt.title('Partial Auto Correlation')
#plt.show()
#
## 一阶差分后序列的自相关，偏自相关
#s1 = pd.Series(data = s_power_consumption.values[0:-1])
#s2 = pd.Series(data = s_power_consumption.values[1:])
#delta_power = s2-s1
#
#corr = auto_corr(delta_power,range(1,60))
#plt.stem(corr)
#plt.title('Auto Correlation')
#plt.show()
#
#partial_coefs = partial_corrs(delta_power.values, lag=30)
#plt.stem(partial_coefs)
#plt.title('Partial Auto Correlation')
#plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(s_power_consumption.values,freq=14)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid


plt.rc("figure", figsize=(25, 10))
plt.subplot(411)
plt.plot(s_power_consumption.values, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

partial_coefs = partial_corrs(residual[7:-8], lag=200)
plt.stem(partial_coefs)
plt.title('Partial Auto Correlation')
plt.show()

def corr(s, k):
    """
    求时间序列相关系数lag=k；
    可视化
    """
    n = len(s)
    x = []; y = []
    for i in range(0,n-k):
        x.append([s[i]])
        y.append([s[i+k]])
    
    # least square by myself
    x = np.array(x)
    y = np.array(y)
    one = np.ones((x.shape[0],1))
    x = np.concatenate((one,np.array(x)),axis=1)
    coefs = np.dot(pinv(x),y)
    coef = coefs[1]

    return coef

def auto_corr(s, lags):
    return np.array([corr(s, k) for k in lags])

corr = auto_corr(residual[7:-8],range(1,200))
plt.stem(corr)
plt.title('Auto Correlation')
plt.show()
