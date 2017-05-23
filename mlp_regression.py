# -*- coding: utf-8 -*-
"""
Created on Tue May 23 07:26:27 2017

@author: Young
"""
import numpy as np
import matplotlib.pyplot as plt

# scaling
fr_x = open('feature.csv','r')
X_list = []
for line in fr_x:
    X_list.append(line.strip().split(','))
    
window_size = 60
prediction_period = 30

#X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))