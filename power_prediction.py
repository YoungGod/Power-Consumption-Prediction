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
#    return df.groupby('record_date')['power_consumption'].sum()
    s_power = df.groupby('record_date')['power_consumption'].sum()
    selection1 = pd.date_range('2015-02-12',periods=14)
    selection2 = pd.date_range('2016-02-4',periods=14)
    return s_power.drop(selection1).drop(selection2)

def load_new_data():
    df = pd.read_csv('Tianchi_power_9.csv')
    df['record_date'] = pd.to_datetime(df['record_date'])
    power9 = df.groupby('record_date')['power_consumption'].sum()
    power = load_data()
    return pd.concat((power,power9))

def load_weathter():
    df = pd.read_csv('weather.csv')
    ave_t = (df.t_max+df.t_min)/2
#    return smooth(ave_t.values,7)
    return ave_t.values

def create_weather(seq, input_lags, pre_period):
    """
    功能：根据时间序列array，及给定的输入时滞及预测时长，构建训数据集(X
    """
    X = []
    n = len(seq)
    window = input_lags + pre_period
    for i in xrange(n - window + 1):
        # if do like this, you need to pay attention
        x = seq[i + window-pre_period: window + i]
        X.append(x)
    return np.array(X)
    
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

def deseasonal_add(seq, freq=7):
    decomposition = seasonal_decompose(seq, model='additive', freq=freq)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    return trend, seasonal, residual

def deseasonal_mul(seq, freq=7):
    decomposition = seasonal_decompose(seq, model='multiplicative', freq=freq)
    return decomposition.trend, decomposition.seasonal, decomposition.resid

def err_evaluation(y_pred,y):
    return sum(((y_pred-y)**2).mean(axis = 1))


def corr(s, k):
    """
    求时间序列相关系数lag=k；
    """
    n = len(s)
    x = []; y = []
    for i in range(0,n-k):
        x.append([s[i]])
        y.append([s[i+k]])
    
    # least square by myself
    x = np.array(x)
    y = np.array(y)
    one = np.ones((x.shape[0],1))
    x = np.concatenate((one,np.array(x)),axis=1)
    coefs = np.dot(np.linalg.pinv(x),y)
    coef = coefs[1]

    return coef

def auto_corr(s, lags):
    """
    时间序列自相关函数
    """
    return np.array([corr(s, k) for k in lags])

def plot_auto_corr(s, lags=range(1,200)):
    """
    绘制自相关函数
    """
    corr_coefs = auto_corr(s,lags)
    plt.figure()
    plt.stem(corr_coefs)
    plt.title('Auto Correlation')
    return plt

def partial_corr(s, k):
    """
    第k阶偏自相关
    w0 + w(k)*x(1) +w(k-1)*x(2) + ... + w(1)*x(k) = x(k+1)
    """
    n = len(s)
    X = [];Y=[]
    for i in range(0,n-k-1):
        X.append(s[i:i+k])
        Y.append(s[i+k+1])
    X = np.array(X); Y = np.array(Y)
    one = np.ones((X.shape[0],1))
    X = np.concatenate((one,X), axis=1)
    coef = np.dot(np.linalg.pinv(X),Y)

    return coef[1]  # 注意取参数的位置w(k),距离x(k+1)相距k的项：w(k)*x(1)
       
def partial_corrs(s, lags=range(1,100)):
    """
    偏自相关函数
    """
    return np.array([partial_corr(s, k) for k in lags])

def plot_partial_corr(s, lags=range(1,100)):
    """
    绘制偏自相关函数
    """
    partial_coefs = partial_corrs(s,lags)
    plt.figure()
    plt.stem(partial_coefs)
    plt.title('Partial Correlation')
    return plt


def write_result(y,path='Tianchi_power_predict_table.csv'):
    """
    # write to file
    """
    fr = open(path,'w')
    fr.write('record_date,power_consumption\n')
    for i,power in enumerate(y):
        if i+1 < 10:
            fr.write('2016100%s,'%(i+1)+str(int(power))+'\n')
        else:
            fr.write('201610%s,'%(i+1)+str(int(power))+'\n')
    fr.close()

def plot_learning_curve(estimator, title, X, y,
                        ylim=None, cv=None, scoring=None,
                        n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve
    
    Parameters
    ----------
    estimator: object type that implements the "fit" and "predict" methods.
    title: string; title for the chart.
    X: traning vector, shape (n_samples, n_features)
    y: target, shape (n_samples,)
    ylim: tuple, shape (ymin, ymax)
          Defines minimum and maximum yvalues plotted.
    cv: int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the dafault 3-fold cross-validation
          - Interger, to specify the number of folds
          - An object to be used as a cross-validation generator
    """
    from sklearn.model_selection import learning_curve
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, 
            train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt,train_sizes

def smooth(y, box_pts):
    """
    简单的平滑滤波
    ref:
    https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


if __name__ == "__main__":
#    # load_data()
#    power = load_data()
#    power.plot()
#    
#    # load_new_data()
#    new_power = load_new_data()
#    new_power.plot()
#    
#    # create_dataset()
#    seq = new_power.values
#    input_lags = 30
#    pre_period = 30
#    X, Y = create_dataset(seq, input_lags, pre_period)
#    fig, ax = plt.subplots()
#    ax.plot(Y[-1])
#    ax.plot(X[-1])
#    
#    y = np.array([1,2,3])
##    write_result(y,path='Tianchi_power_predict_table_test.csv')
#    
#
#    # plot_learning _curve()
#    from sklearn.naive_bayes import GaussianNB
#    from sklearn.svm import SVC
#    from sklearn.datasets import load_digits
#    from sklearn.model_selection import ShuffleSplit
#    digits = load_digits()
#    X, y = digits.data, digits.target
#    
#    title = "Learning Curves (Naive Bayes)"
#    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
#    estimator = GaussianNB()
#    plot_learning_curve(estimator, title, X, y, ylim=(0.7,1.01),
#                        cv=cv, scoring='accuracy')
#    plt,train_sizes = plot_learning_curve(estimator=reg, title='MLP',
#                    X=X, y=Y, cv=30)
    plot_learning_curve(estimator=reg, title='MLP',
                    X=X, y=Y, cv=30,scoring='neg_mean_squared_error')
    
#    title = "Learing Curves (SVM, RBF kernel, $\gamma=0.001$)"
#    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
#    estimator = SVC(gamma=0.001)
#    plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv)
#    
#    # plot_partial_corr()
#    plot_partial_corr(np.array(range(100)),lags=range(1,50))
#    plot_auto_corr(np.array(range(100)),lags=range(1,50))


























