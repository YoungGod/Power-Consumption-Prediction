# -*- coding: utf-8 -*-
"""
Created on Mon May 29 12:44:23 2017

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
#s_power_consumption.index = pd.to_datetime(s_power_consumption.index).sort_values()

from statsmodels.tsa.arima_model import ARIMA
period = 60
s = s_power_consumption.values[:-period]
test = s_power_consumption.values[-period:]
s = s.astype(np.float64)
#model = ARIMA(s, (20,1,10)).fit()
#pred = model.forecast(period)[0]
#plt.plot(pred,label='predict')
#plt.plot(test,label='real')
#plt.legend()
#plt.show()
#
#err = abs(pred - test)/test
#plt.plot(err)
#plt.title('error')

# score
def score(pred,test):
    err = abs(pred - test)/test
    return err.sum()

# 暴利搜索参数 d,p,q
models = []
for d in xrange(0,3):
    for p in xrange(0,31):
        for q in xrange(0,31):
            try:
                model = ARIMA(s, (p,d,q)).fit()
                pred = model.forecast(period)[0]
                models.append((score(pred,test),p,d,q))
            except:
                continue
            
models.sort()
best_score, p, d, q = models[0]
print 'best model:', models[0]
model = ARIMA(s, (p,d,q)).fit()
pred = model.forecast(period)[0]
plt.plot(pred,label='predict')
plt.plot(test,label='real')
plt.legend()
plt.show()

err = abs(pred - test)/test
plt.plot(err)
plt.title('error')



