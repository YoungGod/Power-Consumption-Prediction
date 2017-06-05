# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:47:27 2017

@author: Young
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

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

# df for dataframe, s for series
df = pd.read_csv('Tianchi_power.csv')
df['record_date'] = pd.to_datetime(df['record_date'])

# total power consumption
# 先要把record_date格式转换
s_power_consumption = df.groupby('record_date')['power_consumption'].sum()
power = s_power_consumption.values

# create day types
# 2015-1-1 is wendsday so ..
#day_type = ['wen','thu','fri','sat','sun','mon','tue']
day_type = [3,4,5,6,7,1,2]    # for sklearn
day_type = [3,3,3,6,7,1,3]
rest_days = []
if s_power_consumption.size % 7 == 0:
    num_weeks = s_power_consumption.size / 7
else:
    num_rest_days = s_power_consumption.size % 7
    rest_days = day_type[0:num_rest_days]
    
s_day_type = pd.Series(data = day_type * num_weeks + rest_days, index = s_power_consumption.index)

# now, we need do some exploration and analysis of the collected data
# for example, exclude the anomonly days


# scaling the power consumption
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import RobustScaler
#std_sca = StandardScaler().fit(power.reshape(-1,1))
#seq = StandardScaler().fit_transform(power.reshape(-1,1))
#rob_sca = RobustScaler().fit(s_power_consumption.values.reshape(-1,1))
#data_rob = RobustScaler().fit_transform(s_power_consumption.values.reshape(-1,1)).flatten()

# decomposition
from statsmodels.tsa.seasonal import seasonal_decompose    
decomposition = seasonal_decompose(power,freq=7)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

trend = np.concatenate((np.tile(trend[3],3),trend[3:-3],np.tile(trend[-4],3))) # 首尾需要合理填充
residual = power - trend - seasonal

# for trend
input_lags = 40
hidden = 35
pre_period = 30
window_size = input_lags + pre_period

seq = trend
std_sca = StandardScaler().fit(power.reshape(-1,1))
seq = std_sca.transform(np.array(seq).reshape(-1,1))

X, Y = create_dataset(seq.flatten(), input_lags, pre_period)
X_train = X[:-1]; X_test = X[-1]
Y_train = Y[:-1]; Y_test = Y[-1]

reg = RandomForestRegressor(verbose=True,max_features = 'auto',min_samples_split=2)
reg.fit(X_train,Y_train)
pred_y = reg.predict(X_test)

def test_plot(pred,test):
    plt.plot(pred.flatten(),label='predict')
    plt.plot(test.flatten(),label='real')
    plt.legend()
    plt.show()
     
pred = std_sca.inverse_transform(pred_y.reshape(-1,1))
test = std_sca.inverse_transform(Y_test.reshape(-1,1))

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
reg.fit(X_train,Y_train)
 
# new input
window = input_lags + pre_period
x = seq[-window:-window+input_lags]
x = std_sca.transform(np.array(x).reshape(-1,1)).flatten()

pred = reg.predict(x)
pred = std_sca.inverse_transform(pred.reshape(-1,1))

plt.plot(pred.flatten(),label='predict')
plt.legend()
plt.show()