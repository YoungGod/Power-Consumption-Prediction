# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 07:59:43 2017

@author: Young
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor

def create_dataset(seq, input_lags, pre_period):
    """
    功能：根据时间序列，及给定的输入时滞及预测时长，构建训数据集(X,Y)
    """
    X = []; Y = []
    n = len(seq)
    window = input_lags + pre_period
    for i in xrange(n - window + 1):
        # if do like this, you need to pay attention
        x = seq[i: input_lags + i]
        y = seq[input_lags + i: window + i]

        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)
def err_evaluation(y_pred,y):
    return sum(((y_pred-y)**2).mean(axis = 1))


def choose_best_lag(seq, pre_period, lags = range(1,30)):
    """
    选择最佳lazzy model,及输入时滞
    模型：(误差，延时，邻居数)
    """
    models = []
    # 标准化
    std_sca = StandardScaler().fit(np.array(seq).reshape(-1,1))
    seq = std_sca.transform(np.array(seq).reshape(-1,1))
    
    # 根据时滞及序列创建数据集,并进行交叉验证
    from sklearn.model_selection import train_test_split
    for input_lag in lags:
#        window = input_lag + pre_period
        X, Y = create_dataset(seq.flatten(), input_lag, pre_period)
        # do more cv
#        for state in range(0,3):
        err = 0.0
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.01, random_state=0)
        for lag in lags:
            hidden = (lag + pre_period + 3)/2
            reg = MLPRegressor(activation = 'relu',hidden_layer_sizes = (hidden,),
                               max_iter=10000,learning_rate='adaptive',
                               tol=0.0,warm_start=True,solver='adam')
            reg.fit(X_train,y_train)
            
            y_pred = reg.predict(X_test)
            err += err_evaluation(y_pred,y_test)
        models.append((err/len(X_test),lag))
    models.sort()
    best_lag = models[0][1]

    return models, best_lag

# df for dataframe, s for series
df = pd.read_csv('Tianchi_power.csv')
df['record_date'] = pd.to_datetime(df['record_date'])

# total power consumption
# 先要把record_date格式转换
s_power_consumption = df.groupby('record_date')['power_consumption'].sum()
power = s_power_consumption.values


# for example, exclude the anomonly days
day_type = [3,4,5,6,7,1,2]
rest_days = []
if s_power_consumption.size % 7 == 0:
    num_weeks = s_power_consumption.size / 7
else:
    num_rest_days = s_power_consumption.size % 7
    rest_days = day_type[0:num_rest_days]
    
s_day_type = pd.Series(data = day_type * num_weeks + rest_days, index = s_power_consumption.index)

s = pd.Series(data=s_power_consumption.values,index=s_day_type.values)

day_type = [3,4,5,6,7,1,2]
data = []
for day in day_type:
    data.append(s.values[np.where(s.index==day)])
    
data = np.array(data)
df = pd.DataFrame(data = data.T,columns=['Wed','Thu','Fri','Sat','Sun','Mon','Tue',])
s_median = df.median()

median = s_median.values
power = s_power_consumption.values
median_expand = np.tile(median,87)
deseason_power = power - median_expand
power = deseason_power
# scaling the power consumption
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
#std_sca = StandardScaler().fit(power.reshape(-1,1))
#seq = StandardScaler().fit_transform(power.reshape(-1,1))
#rob_sca = RobustScaler().fit(s_power_consumption.values.reshape(-1,1))
#data_rob = RobustScaler().fit_transform(s_power_consumption.values.reshape(-1,1)).flatten()

# decomposition

# for trend
input_lags = 70
pre_period = 30

seq = power
#std_sca = StandardScaler().fit(power.reshape(-1,1))
#seq = std_sca.transform(np.array(seq).reshape(-1,1))
rob_sca = RobustScaler().fit(power.reshape(-1,1))
seq = rob_sca.transform(np.array(seq).reshape(-1,1))
#models, input_lags = choose_best_lag(seq, pre_period, lags = range(20,100))
hidden = (input_lags + pre_period + 3)/2
window_size = input_lags + pre_period

X, Y = create_dataset(seq.flatten(), input_lags, pre_period)

reg = MLPRegressor(activation = 'relu',hidden_layer_sizes = (hidden,),
                               max_iter=10000,verbose=True,learning_rate='adaptive',
                               tol=0.0,warm_start=True,solver='adam')




X_train = X[:-1]; X_test = X[40]
Y_train = Y[:-1]; Y_test = Y[40]

reg = MLPRegressor(activation = 'relu',hidden_layer_sizes = (hidden,),
                               max_iter=10000,verbose=True,learning_rate='adaptive',
                               tol=0.0,warm_start=True,solver='adam')
reg.fit(X_train,Y_train)
pred_y = reg.predict(X_test)

def test_plot(pred,test):
    plt.plot(pred.flatten(),label='predict')
    plt.plot(test.flatten(),label='real')
    plt.legend()
    plt.show()
     
pred = rob_sca.inverse_transform(pred_y.reshape(-1,1))
test = rob_sca.inverse_transform(Y_test.reshape(-1,1))

# drawing
test_plot(pred,test)

err = abs(pred-test)/test
plt.plot(err,label='err')
plt.legend()
plt.show()

# 误差方差
re_err = abs(pred-test)
mean_fit_err = abs(reg.predict(X_train)-Y_train).sum().mean()
mean_pre_err = re_err.mean()

print 'fit err:', mean_fit_err
print 'pre err', mean_pre_err      


"""
Final prediction
"""
# final prediction
X_train = X
Y_train = Y
# new input
window = input_lags + pre_period
x = seq[-window:-window+input_lags]
x = rob_sca.transform(np.array(x).reshape(-1,1)).flatten()
pred = 0
states = range(0,30)
for state in states:
    reg = MLPRegressor(activation = 'relu',hidden_layer_sizes = (hidden,),
                                   max_iter=10000,verbose=False,learning_rate='adaptive',
                                   tol=0.0,warm_start=True,solver='adam',random_state=state)
    reg.fit(X_train,Y_train)
    pred += reg.predict(x)
pred = pred/len(states)

pred = rob_sca.inverse_transform(pred.reshape(-1,1))
pred = pred.flatten() + median_expand[0:pre_period]

plt.plot(pred.flatten(),label='predict')
plt.legend()
plt.show()