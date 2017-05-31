# -*- coding: utf-8 -*-
"""
Created on Tue May 30 03:07:11 2017

@author: Young
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor

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
day_type = [3,4,5,6,7,1,2]    # for sklearn
day_type = [3,3,3,6,7,1,3]
day_type = [3,3,3,6,7,3,3]
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

## pre 
#data_rob = np.concatenate((data_rob[0:121],data_rob[180:]))
#s_day_type = pd.Series(data = s_day_type.values)
#s_day_type.drop(range(121,180))

# creat samples
# feature X, and target Y
# the month sep has 30 days so, target y is an vector with 30 dimensions
# here, we use the previous 30 days power and day types plus the next 30 day types to predict
# the next 30 day power 
input_size = 90
input_sizes = [30,45,60,75,90,105,120,135,150]
input_sizes = [90,120,150]
random_states = range(0,2)
hiddens = np.linspace(90,300,10).astype(np.int)
hiddens = [90,120,150,180,300]
prediction_period = 30

# score
def score(pred,test):
    pred = rob_sca.inverse_transform(pred.reshape(-1,1))
    test = rob_sca.inverse_transform(test.reshape(-1,1))
    err = abs(pred - test)/test
    return err.sum()

## chosing the best model
#models = []
#
#for input_size in input_sizes:
#  
#    window_size = input_size + prediction_period
#    
#    #seq_length = s_power_consumption.size
#    seq_length = data_rob.size
#    
#    X_power = []
#    XY_day_type = []
#    Y_power = []
#    
#    # 构建数据集
#    for i in xrange(0,seq_length-window_size):
#        xy_power = data_rob[i:window_size+i]
#        x_power = xy_power[0:window_size-prediction_period]
#        X_power.append(x_power)
#        y_power = xy_power[-prediction_period:]
#        Y_power.append(y_power)
#        
#        xy_day_type = s_day_type.values[i:window_size+i]
#        XY_day_type.append(xy_day_type)
#            
#    # training and test set
#    X_power = np.array(X_power)
#    XY_day_type = np.array(XY_day_type)
#    X = np.concatenate((X_power,XY_day_type),axis = 1)
#    
#    # One hot coding
#    enc = OneHotEncoder(categorical_features=np.arange(window_size-prediction_period,X.shape[1]))
#    X = enc.fit_transform(X)
#    
#    Y = np.array(Y_power)
#    
#    # the last month for testing
#    X = X.toarray()
##    X_train = X[:-30]; X_test = X[-30:]
##    Y_train = Y[:-30]; Y_test = Y[-30:]
#        
#    for hidden in hiddens:
#        s_score = 0
#        for state in random_states:
#            reg = MLPRegressor(activation = 'relu',hidden_layer_sizes = (hidden,30),
#                               max_iter=10000,verbose=False,learning_rate='adaptive',
#                               tol=0.0,warm_start=True,solver='adam',random_state=state)
#            for i in xrange(0,5):
#                X_train = X[:-30+i]; X_test = X[-30+i]
#                Y_train = Y[:-30+i]; Y_test = Y[-30+i]          
#                reg.fit(X_train,Y_train)
#                pred_y = reg.predict(X_test.reshape(1,-1))
#                s_score += score(pred_y,Y_test)
##            reg.fit(X_train,Y_train)
##            pred_y = reg.predict(X_test)
##            s_score += score(pred_y,Y_test)
#        models.append((s_score/len(random_states),input_size,hidden))
#
## best model
#models.sort()
#best_score, input_size, hidden = models[0]

input_size = 120
hidden = 300

reg = MLPRegressor(activation = 'relu',hidden_layer_sizes = (hidden,30),
                               max_iter=10000,verbose=True,learning_rate='adaptive',
                               tol=0.0,warm_start=True,solver='adam')
window_size = input_size + prediction_period

#seq_length = s_power_consumption.size
seq_length = data_rob.size

X_power = []
XY_day_type = []
Y_power = []

# 构建数据集
for i in xrange(0,seq_length-window_size):
    xy_power = data_rob[i:window_size+i]
    x_power = xy_power[0:input_size]
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
enc = OneHotEncoder(categorical_features=np.arange(window_size-prediction_period,X.shape[1]))
X = enc.fit_transform(X)

Y = np.array(Y_power)

# the last month for testing
X = X.toarray()
X_train = X[:-1]; X_test = X[-1]
Y_train = Y[:-1]; Y_test = Y[-1]
reg.fit(X_train,Y_train)
pred_y = reg.predict(X_test)

def test_plot(pred,test):
    plt.plot(pred.flatten(),label='predict')
    plt.plot(test.flatten(),label='real')
    plt.legend()
    plt.show()
     
pred = std_sca.inverse_transform(pred_y.reshape(-1,1))
test = std_sca.inverse_transform(Y_test.reshape(-1,1))
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


# final prediction the 9th month 120,90,30
X_train = X
Y_train = Y

day_type9 = [3,3,3,6,7,3,3]    # for sklearn
rest_days = []
num_weeks = 30 / 7
if 30 % 7 != 0:
    num_rest_days = 30 % 7
    rest_days = day_type[0:num_rest_days]
    
s_day_type9 = pd.Series(data = day_type9 * num_weeks + rest_days)

x9_power = data_rob[-(window_size-prediction_period):]
x9_day_type = s_day_type.values[-(window_size-prediction_period):]
x9 = np.concatenate((x9_power,x9_day_type,s_day_type9.values))
x9 = enc.transform(x9)
power9 = 0
for state in range(0,30):

    reg = MLPRegressor(activation = 'relu',hidden_layer_sizes = (hidden,30),
                       max_iter=10000,verbose=False,learning_rate='adaptive',
                       tol=0.0,warm_start=True,solver='adam',random_state=state)

    reg.fit(X_train,Y_train)     
    power9 += reg.predict(x9)
    

power9 = rob_sca.inverse_transform(power9.reshape(-1,1)/30)

# write to file
fr = open('Tianchi_power_predict_table.csv','w')
fr.write('record_date,power_consumption\n')
for i,power in enumerate(power9):
    if i+1 < 10:
        fr.write('2016090%s,'%(i+1)+str(int(power))+'\n')
    else:
        fr.write('201609%s,'%(i+1)+str(int(power))+'\n')
fr.close()

plt.plot(power9)