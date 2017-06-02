# -*- coding: utf-8 -*-
"""
Created on Tue May 23 05:32:41 2017

@author: Young
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df for dataframe, s for series
df = pd.read_csv('Tianchi_power.csv')
df['record_date'] = pd.to_datetime(df['record_date'])

# total power consumption
# 先要把record_date格式转换
s_power_consumption = df.groupby('record_date')['power_consumption'].sum()
#pivoted = df.pivot('record_date','user_id','power_consumption')
#s_power_consumption = pivoted[144]
s_power_consumption.index = pd.to_datetime(s_power_consumption.index).sort_values()

# create day types
# 2015-1-1 is wendsday so ..
#day_type = ['wen','thu','fri','sat','sun','mon','tue']
day_type = [3,3,3,6,7,3,3]    # for sklearn
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

# pre 
data_rob = np.concatenate((data_rob[0:121],data_rob[180:]))
s_day_type = pd.Series(data = s_day_type.values)
s_day_type.drop(range(121,180))
# creat samples
# feature X, and target Y
# the month sep has 30 days so, target y is an vector with 30 dimensions
# here, we use the previous 30 days power and day types plus the next 30 day types to predict
# the next 30 day power 
window_size = 120
prediction_period = 30
input_size = window_size - prediction_period
#seq_length = s_power_consumption.size
seq_length = data_rob.size

X_power = []
XY_day_type = []
Y_power = []

#fr_x = open('feature.csv','w')
#fr_y = open('target.csv','w')
for i in xrange(0,seq_length-window_size+1):
    xy_power = data_rob[i:window_size+i]
    x_power = xy_power[0:window_size-prediction_period]
    X_power.append(x_power)
    y_power = xy_power[-prediction_period:]
    Y_power.append(y_power)
    
    xy_day_type = s_day_type.values[i:window_size+i]
    XY_day_type.append(xy_day_type)
    
#    for power in x_power:
#        fr_x.write(str(power)+',')
#    for i in xrange(0,window_size-1):
#        fr_x.write(str(xy_day_type[i])+',')  # for sklearn
#    fr_x.write(str(xy_day_type[i])+'\n')
##        fr_x.write(xy_day_type[i]+',')
##    fr_x.write(xy_day_type[-1]+'\n')
#    
#    for i in xrange(0,prediction_period-1):
#        fr_y.write(str(y_power[i])+',')
#    fr_y.write(str(y_power[-1])+'\n')
#fr_x.close()
#fr_y.close()   

# training and test set
X_power = np.array(X_power)
XY_day_type = np.array(XY_day_type)
X = np.concatenate((X_power,XY_day_type),axis = 1)

# One hot coding
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(categorical_features=np.arange(window_size-prediction_period,X.shape[1]))
X = enc.fit_transform(X)
#new_sca = StandardScaler(with_mean=False)
#X = .fit_transform(X)

Y = np.array(Y_power)

# the last month for testing
X = X.toarray()
X_train = X[:-30]; X_test = np.concatenate((X[-60].reshape(1,-1),X[-30].reshape(1,-1)))
Y_train = Y[:-30]; Y_test = np.concatenate((Y[-60].reshape(1,-1),Y[-30].reshape(1,-1)))

from sklearn.neural_network import MLPRegressor

hidden = input_size+3*input_size+3*prediction_period
reg = MLPRegressor(activation = 'relu',hidden_layer_sizes = (600,30),
                   max_iter=10000,verbose=True,learning_rate='adaptive',
                   tol=0.0,warm_start=True,solver='adam')

reg.fit(X_train,Y_train)

pred_y = reg.predict(X_test)


plt.plot(pred_y.flatten(),label='predict')
plt.plot(Y_test.flatten(),label='real')
plt.legend()
plt.show()


pred = rob_sca.inverse_transform(pred_y.reshape(-1,1))
test = rob_sca.inverse_transform(Y_test.reshape(-1,1))


err = abs(pred-test)/test

plt.plot(pred.flatten(),label='predict')
plt.plot(test.flatten(),label='real')
plt.legend()
plt.show()

plt.plot(err,label='err')
plt.legend()
plt.show()

# 误差方差
re_err = abs(pred-test)
mean_fit_err = abs(reg.predict(X_train)-Y_train).sum().mean()
mean_pre_err = re_err.mean()

print 'fit err:', mean_fit_err
print 'pre err', mean_pre_err      


## final prediction the 9th month 120,90,30
#X_train = X
#Y_train = Y
#
#from sklearn.neural_network import MLPRegressor
#
#reg = MLPRegressor(activation = 'relu',hidden_layer_sizes = (120,30),
#                   max_iter=10000,verbose=True,learning_rate='adaptive',
#                   tol=0.0,warm_start=True,solver='adam',random_state=0)
#
#reg.fit(X_train,Y_train)
#
## write to file
#day_type9 = [3,3,3,6,7,3,3]    # for sklearn
#rest_days = []
#num_weeks = 30 / 7
#if 30 % 7 != 0:
#    num_rest_days = 30 % 7
#    rest_days = day_type[0:num_rest_days]
#    
#s_day_type9 = pd.Series(data = day_type9 * num_weeks + rest_days)
#
#x9_power = data_rob[-(window_size-prediction_period):]
#x9_day_type = s_day_type.values[-(window_size-prediction_period):]
#x9 = np.concatenate((x9_power,x9_day_type,s_day_type9.values))
#
#x9 = enc.transform(x9)
#
#power9 = reg.predict(x9) 
#
#power9 = rob_sca.inverse_transform(power9.reshape(-1,1))
#
#fr = open('Tianchi_power_predict.csv','w')
#fr.write('record_date,power_consumption\n')
#for i,power in enumerate(power9):
#    if i+1 < 10:
#        fr.write('2016090%s,'%(i+1)+str(int(power))+'\n')
#    else:
#        fr.write('201609%s,'%(i+1)+str(int(power))+'\n')
#fr.close()
#
#plt.plot(power9)