# -*- coding: utf-8 -*-
"""
Created on Fri Jun 09 05:34:38 2017

@author: Young
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 15:49:23 2017

@author: Young
"""
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from power_prediction import *

"""
8,9月数据用于test，其余数据用于训练集交叉验证
"""
# load power data
#s_power = load_data()
s_power = load_new_data()
power = s_power.values
std_sca = StandardScaler().fit(power.reshape(-1,1))
power = std_sca.transform(power.reshape(-1,1)).flatten()

# deseasonal
trend, season, resi = deseasonal_add(power, freq=7)
seq = power - season  # predicting the (trend + residual)

# prediction period
pre_period = 31
# feature pool 200 lags
input_lags = 60   

# split training and testing sequence
seq_train = seq[:-1*pre_period]
seq_test = seq[-1*pre_period:]  # 8,9月数据用于testing
seq_test = std_sca.inverse_transform(seq_test.reshape(-1,1))

# create training dataset
X, Y = create_dataset(seq_train, input_lags, pre_period)

# set the overall random_state
rand_state = 1
# MLP regressor
hidden = (input_lags + pre_period) / 2
reg = MLPRegressor(activation = 'relu',hidden_layer_sizes = (hidden,),
                               max_iter=10000,verbose=False,
                               learning_rate='adaptive',
                               tol=0.0,random_state=1,solver='adam')
# cv
kf = KFold(n_splits = 3)
# GirdSearchCV
param_grid = dict(alpha = 10.0**-np.arange(-2,0))
regs = GridSearchCV(reg, param_grid, cv=kf)
regs.fit(X, Y)
# chose the best estimator
reg = regs.best_estimator_

# predict
#X_test = np.reshape(seq[-1*input_lags:],(2,-1))  # 8,9 inputs
X_test = seq[-1*input_lags:].reshape(1,-1)
y_trend = reg.predict(X_test)
y = y_trend.flatten() + season[-1*pre_period:]
y = std_sca.inverse_transform(y.reshape(-1,1))

# draw the results
fig, ax = plt.subplots()
ax.plot(y.flatten(),label='prediction')
ax.plot(seq_test,label='real')
ax.legend()
plt.show()

# write to file
#write_result(y)