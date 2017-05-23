# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:47:27 2017

@author: Young
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df for dataframe, s for series
df = pd.read_csv('Tianchi_power.csv')
# df['record_date'] = pd.to_datetime(df['record_date'])

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
    
s_day_type = pd.Series(data = day_type * num_weeks + rest_days, index = s_power_consumption.index)

# now, we need do some exploration and analysis of the collected data
# for example, exclude the anomonly days


# scaling the power consumption
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
std_sca = StandardScaler().fit(s_power_consumption.values.reshape(-1,1))
data_std = StandardScaler().fit_transform(s_power_consumption.values.reshape(-1,1)).flatten()
rob_sca = RobustScaler().fit(s_power_consumption.values.reshape(-1,1))
data_rob = RobustScaler().fit_transform(s_power_consumption.values.reshape(-1,1)).flatten()

# creat samples
# feature X, and target Y
# the month sep has 30 days so, target y is an vector with 30 dimensions
# here, we use the previous 30 days power and day types plus the next 30 day types to predict
# the next 30 day power 
window_size = 25
prediction_period = 30
seq_length = s_power_consumption.size

X_power = []
XY_day_type = []
Y_power = []

for i in xrange(0,seq_length-window_size):
    xy_power = data_std[i:window_size+i]
    x_power = xy_power[0:prediction_period]
    X_power.append(x_power)
    y_power = xy_power[-prediction_period:]
    Y_power.append(y_power)
    
    xy_day_type = s_day_type.values[i:window_size+i]
    XY_day_type.append(xy_day_type)
    
# training and test set
X_power = np.array(X_power)
XY_day_type = np.array(XY_day_type)
X = np.concatenate((X_power,XY_day_type),axis = 1)

# One hot coding
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(categorical_features=np.arange(30,X.shape[1]))
X = enc.fit_transform(X)

Y = np.array(Y_power)

# the last month for testing
X = X.toarray()
X_train = X[:-1]; X_test = X[-1:]
Y_train = Y[:-1]; Y_test = Y[-1:]

from sklearn.neural_network import MLPRegressor

reg = MLPRegressor(activation = 'logistic',hidden_layer_sizes = (100,30),
                   max_iter=10000,verbose=True,learning_rate='adaptive',
                   tol=0.0,warm_start=True)
reg.fit(X_train,Y_train)

pred_y = reg.predict(X_test)


plt.plot(pred_y.flatten(),label='predict')
plt.plot(Y_test.flatten(),label='real')
plt.legend()
plt.show()


pred = rob_sca.inverse_transform(pred_y.reshape(-1,1))
test = rob_sca.inverse_transform(Y_test.reshape(-1,1))

re_err = abs(pred-test)/test

plt.plot(pred.flatten(),label='predict')
plt.plot(test.flatten(),label='real')
plt.legend()
plt.show()

plt.plot(re_err,label='err')
plt.legend()
plt.show()





