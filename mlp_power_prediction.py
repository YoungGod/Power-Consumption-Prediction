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

# load power data
s_power = load_data()
power = s_power.values
std_sca = StandardScaler().fit(power.reshape(-1,1))
power = std_sca.transform(power.reshape(-1,1)).flatten()

trend, season, resi = add_deseason(power, freq=7)
seq = power - season

# prediction period
pre_period = 30
# feature pool 200 lags
input_lags = 60   

# create dataset
X, Y = create_dataset(seq, input_lags, pre_period)

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
param_grid = dict(alpha = 10.0**-np.arange(1,7))
regs = GridSearchCV(reg, param_grid)
regs.fit(X, Y)

reg = regs.best_estimator_
x = power[-input_lags:]
y = reg.predict(x.reshape(1,-1))
y = y + season[0:30]
y = std_sca.inverse_transform(y.reshape(-1,1))
fig, ax = plt.subplots()
ax.plot(y.flatten())
ax.plot(power9)

























