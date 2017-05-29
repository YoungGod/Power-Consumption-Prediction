# -*- coding: utf-8 -*-
"""
Created on Wed May 24 05:42:21 2017

@author: Young
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df for dataframe, s for series
df = pd.read_csv('Tianchi_power.csv')
df['record_date'] = pd.to_datetime(df['record_date'])

# total power consumption
s_power_consumption = df.groupby('record_date')['power_consumption'].sum()
s_power_consumption.index = pd.to_datetime(s_power_consumption.index).sort_values()

# create day types
# 2015-1-1 is wendsday so ..
#day_type = ['wen','thu','fri','sat','sun','mon','tue']
day_type = [3,4,5,6,7,1,2]    # for sklearn
rest_days = []
if s_power_consumption.size % 7 == 0:
    num_weeks = s_power_consumption.size / 7
else:
    num_rest_days = s_power_consumption.size % 7
    rest_days = day_type[0:num_rest_days]
    
s_day_type = pd.Series(data = day_type * num_weeks + rest_days, index = s_power_consumption.index,name='day_type')
#s_day_type.rename('day_type')

#dataset = pd.concat([s_power_consumption,s_day_type],axis=1)

# 剔除趋势因子，移动平均
def auto_corr(x, l):
    return [x.autocorr(i) for i in l]

for window_size in range(36,41):
#    window_size = 7
    avg_power = s_power_consumption.rolling(window=window_size,center=False).mean()
    s_power = s_power_consumption - avg_power
    corr = auto_corr(s_power,range(1,60))
    plt.plot(corr,label='windowsize%d'%window_size)
plt.legend()
plt.show()

corr = auto_corr(s_power_consumption,range(1,60))
plt.plot(corr,label='normal')
plt.legend()
plt.show()

# 序列平稳化MA
avg_power = s_power_consumption.rolling(window=30,center=False).mean()
s_power = s_power_consumption - avg_power
s_values = s_power.values[30:]

# 序列平稳化，一阶差分
s1 = pd.Series(data = s_power_consumption.values[0:-1])
s2 = pd.Series(data = s_power_consumption.values[1:])

delta_power = s1-s2
corr = auto_corr(delta_power,range(1,60))
plt.plot(corr,label='normal')
plt.legend()
plt.show()

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(avg_power.values[30:]).show()
plot_acf(s_power_consumption).show()
plot_acf(s_values).show()
plot_acf(delta_power).show()
plt.show()
## 平稳性检验
#from statsmodels.tsa.stattools import adfuller as ADF
#ADF(s_power_consumption)
#ADF(avg_power.values[30:])
#ADF(s_values)
#ADF(delta_power)
#
## 白噪声检验
#from statsmodels.stats.diagnostic import acorr_ljungbox
#acorr_ljungbox(s_power_consumption, lags=1)
#acorr_ljungbox(avg_power.values[30:], lags=1)
#acorr_ljungbox(s_values, lags=1)
#acorr_ljungbox(delta_power, lags=1)
#
#
## 序列平稳化MA
#avg_power = s_power_consumption.rolling(window=5,center=False).mean()
#s_power = s_power_consumption - avg_power
#s_values = s_power.values[6:]
#ADF(s_values)
#acorr_ljungbox(s_values, lags=1)

from statsmodels.tsa.arima_model import ARIMA
s = s_power_consumption.values[:]
s = s.astype(np.float64)
model = ARIMA(s, (6,1,2)).fit()
pred = model.forecast(30)[0]
plt.plot(pred,label='predict')
plt.plot(s[-30:],label='real')
plt.legend()
plt.show()

#def auto_corr(x, lags=1):
#    n = len(x)
#    x = np.array(x)
#    variance = x.var()
#    x = x - x.mean()
#    result = np.correlate(x, x, mode = 'full')[-n+1:-n+lags+1]/\
#                (variance*np.arange(n-1, n-1-lags,-1))
#    return result


# 
pivoted = df.pivot('record_date','user_id','power_consumption')
user = pivoted.mean()
index = user.index
index1 = index[np.where(user.values==1)]


