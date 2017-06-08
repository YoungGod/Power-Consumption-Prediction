# -*- coding: utf-8 -*-
"""
Created on Thu Jun 08 12:28:21 2017

@author: Young
"""

"""
=================================
Common Functions for power prediction
=================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose    

def load_data():
    df = pd.read_csv('Tianchi_power.csv')
    df['record_date'] = pd.to_datetime(df['record_date'])
    return df.groupby('record_date')['power_consumption'].sum()

def load_new_data():
    df = pd.read_csv('Tianchi_power_9.csv')
    df['record_date'] = pd.to_datetime(df['record_date'])
    power9 = df.groupby('record_date')['power_consumption'].sum()
    power = load_data()
    return pd.concat((power,power9))

def create_dataset(seq, input_lags, pre_period):
    """
    功能：根据时间序列array，及给定的输入时滞及预测时长，构建训数据集(X,Y)
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

def add_deseason(seq, freq=7):
    decomposition = seasonal_decompose(seq, freq=freq)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    return trend, seasonal, residual

def err_evaluation(y_pred,y):
    return sum(((y_pred-y)**2).mean(axis = 1))



if __name__ == "__main__":
    # load_data()
    power = load_data()
    power.plot()
    
    # load_new_data()
    new_power = load_new_data()
    new_power.plot()
    
    # create_dataset()
    seq = new_power.values
    input_lags = 30
    pre_period = 30
    X, Y = create_dataset(seq, input_lags, pre_period)
    fig, ax = plt.subplots()
    ax.plot(Y[-1])
    ax.plot(X[-1])
    
    