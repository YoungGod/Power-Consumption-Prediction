# -*- coding: utf-8 -*-
"""
Created on Sat Jun 03 15:20:44 2017

@author: Young
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Tianchi_power_predict_table_mean.csv')

power9 = df['ann_lazzy'].values

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