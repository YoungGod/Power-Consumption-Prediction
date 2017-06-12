# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 08:13:34 2017

@author: Young
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from power_prediction import *

# load power data
#s_power = load_data()
s_power = load_new_data()
#power = s_power.values[:-30] # predict 7th month
power = s_power.values
std_sca = StandardScaler().fit(power.reshape(-1,1))
power_scaled = std_sca.transform(power.reshape(-1,1)).flatten()
#seq = power_scaled
# deseasonal
trend, season, resi = deseasonal_add(power_scaled, freq=7)
seq = power_scaled - season
# denoise
#seq = smooth(seq,3)

# prediction period
pre_period = 31
# feature pool 200 lags
input_lags = 120
# create dataset
seq_train = seq[0:-pre_period]
X, Y = create_dataset(seq_train, input_lags, pre_period)

from sklearn.decomposition import PCA
pca = PCA(n_components=60)
pca.fit(X)
X = pca.transform(X)

# set the overall random_state
rand_state = 1

# RF regressor
reg = RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=None,
                            min_samples_split=2, min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_features='auto', max_leaf_nodes=None,
                            min_impurity_split=1e-07, bootstrap=True,
                            oob_score=False, n_jobs=1,
                            random_state=None, verbose=0,
                            warm_start=False)
# cv
kfcv = KFold(n_splits = 10)
#tscv = TimeSeriesSplit(n_splits = 30)
# GirdSearchCV  best_alpha=125.89254117941765
param_grid = dict(n_estimators = [1000])
#param_grid = dict(alpha = 10.0**-np.arange(-4,-2,0.5),
#              hidden_layer_sizes = [30, 40, 45,60])
regs = GridSearchCV(reg, param_grid, cv=10, scoring='neg_mean_squared_error')
regs.fit(X, Y)

reg = regs.best_estimator_
## learning curve
#plot_learning_curve(estimator=reg, title='MLP',
#                    X=X, y=Y, cv=30,scoring='neg_mean_squared_error')

x = seq_train[-input_lags:]
x = pca.transform(x)
y_trend = reg.predict(x.reshape(1,-1))
y = y_trend + season[-pre_period:]
#y =  y_trend
y = std_sca.inverse_transform(y.reshape(-1,1))

# drawing
fig, ax = plt.subplots()
ax.plot(y.flatten(),label='prediction')
ax.plot(power[-pre_period:],label='real')
ax.legend()

fig, ax = plt.subplots()
re_err = 100*(y.flatten()-power[-pre_period:])/power[-pre_period:]
ax.plot(re_err,label='relative error %')
ax.legend()


## model persistence
#import pickle
#mlp = pickle.dumps(reg)
#reg = pickle.loads(mlp)


#"""
#Final Prediction
#"""
## 10th month prediction
#X, Y = create_dataset(seq, input_lags, pre_period)
#reg.fit(X, Y)
#x = seq[-input_lags:]
#y_trend = reg.predict(x.reshape(1,-1))
#y = y_trend + season[-pre_period-3:-3]
#y = std_sca.inverse_transform(y.reshape(-1,1))
#fig, ax = plt.subplots()
#ax.plot(y.flatten())
