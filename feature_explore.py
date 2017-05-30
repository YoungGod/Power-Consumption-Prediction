# -*- coding: utf-8 -*-
"""
Created on Tue May 30 03:51:36 2017

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
#pivoted = df.pivot('record_date','user_id','power_consumption')

# create day types
# 2015-1-1 is wendsday so ..
#day_type = ['wen','thu','fri','sat','sun','mon','tue']
#day_type = [3,3,3,6,7,3,3]    # for sklearn
day_type7 = [3,4,5,6,7,1,2]
day_type5 = [3,3,5,6,7,1,2]
day_type4 = [3,3,3,6,7,1,3]
day_type3 = [3,3,3,6,7,3,3]
day_type2 = [3,3,3,3,7,3,3]
day_type1 = [3,3,3,3,3,3,3]

day_types = [day_type7,
            day_type5,
            day_type4,
            day_type3,
            day_type2,
            day_type1]
mean = []; std =[]
for day_type in day_types:
    rest_days = []
    if s_power_consumption.size % 7 == 0:
        num_weeks = s_power_consumption.size / 7
    else:
        num_rest_days = s_power_consumption.size % 7
        rest_days = day_type[0:num_rest_days]
        
    s_day_type = pd.Series(data = day_type * num_weeks + rest_days, index = s_power_consumption.index)
    
    s = pd.concat((s_power_consumption,s_day_type),axis=1)
    mean.append(s.groupby(0)['power_consumption'].mean())
    std.append(s.groupby(0)['power_consumption'].std())
    