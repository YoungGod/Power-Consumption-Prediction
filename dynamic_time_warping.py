# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 20:05:57 2017

@author: Young
参考：http://blog.csdn.net/raym0ndkwan/article/details/45614813
      http://nbviewer.jupyter.org/github/alexminnaar/time-series-classification-and-clustering/blob/master/Time%20Series%20Classification%20and%20Clustering.ipynb
"""
import numpy as np

def DTWDistance(s1, s2):
    """
    功能：计算序列s1与s2的动态时间规整距离、规整成本矩阵
    """
    # 使用字典来存储DTW表，cool（在其他语言中，可以使用一个哈希map和point类来实现）
    DTW = {}
    
    # 初始化
    """示意：
(i, -1)
    inf
    inf             待求解
    inf (1,0) (1,1)
    inf (0,0) (0,1)
     0   inf   inf  inf inf  (-1, j)
(-1,-1)
    """
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW [(-1, -1)] = 0
    
    # 动态规划（从前向后迭代，也可以从后向前递归）
    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = np.sqrt((s1[i] - s2[j])**2)
            DTW[(i, j)] = dist + min(DTW[(i-1, j)], DTW[(i, j-1)], DTW[(i-1, j-1)])
    
    return DTW,DTW[(len(s1)-1, len(s2)-1)]
    #return DTW[(len(s1)-1, len(s2)-1)]


def DTWDistanceWithWindow(s1, s2,w):
    """
    功能：计算序列s1与s2的动态时间规整距离，使用了窗口w
          intuition：it is unlikely for si and sj to be matched if i and j are too far apart.
    """
    DTW={}
    
    w = max(w, abs(len(s1)-len(s2)))
    
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
  
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
		
    return np.sqrt(DTW[len(s1)-1, len(s2)-1])


def euclid_dist(t1, t2):
    """
    功能：计算欧拉序列间的距离
    """
    return np.sqrt(sum((t1 - t2)**2))

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pylab as plt
    
    x=np.linspace(0,50,10)
    ts1=pd.Series(3.1*np.sin(x/1.5)+3.5)
    ts2=pd.Series(2.2*np.sin(x/3.5+2.4)+3.2)
    ts3=pd.Series(0.04*x+3.0)
    
    ts1.plot()
    ts2.plot()
    ts3.plot()
    
    plt.ylim(-2,10)
    plt.legend(['ts1','ts2','ts3'])
    plt.show()
    
    print euclid_dist(ts1, ts2)
    print euclid_dist(ts1, ts3)
    
    DTW1, distance1 = DTWDistance(ts1, ts2)
    DTW2, distance2 = DTWDistance(ts1, ts3)
    
    plt.figure()
    d = np.zeros(shape=(len(ts1),len(ts2)))
    for i,j in DTW1.keys():
        if i >= 0 and j >= 0:
            d[i][j] = DTW1.get((i,j))
    plt.imshow(d)
    
    plt.figure()
    d = np.zeros(shape=(len(ts1),len(ts3)))
    for i,j in DTW2.keys():
        if i >= 0 and j >= 0:
            d[i][j] = DTW2.get((i,j))
    plt.imshow(d)