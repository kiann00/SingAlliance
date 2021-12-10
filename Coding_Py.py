#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load set of python modules
import pandas as pd
import numpy as np
import scipy as sc
import datetime
from datetime import date
import time
import matplotlib.pyplot as plt
import scipy.stats as stats

import asyncio
import websockets
import websocket
import json
import sys
import requests
import random
import hmac
from IPython.display import display

start_ = datetime.datetime(2021, 10, 1,0,0).timestamp()
end_ = datetime.datetime(2021, 10, 31,0,0).timestamp()

ticker_ = 'BTC'
string_ = 'https://ftx.com/api/markets/' + ticker_ +'-PERP/candles?resolution=300&start_time=' + str(int(start_))            + '&end_time=' + str(int(end_))

# GET /markets/{market_name}/candles?resolution={resolution}&start_time={start_time}&end_time={end_time}
historical = requests.get(string_).json()
res_btc = pd.DataFrame(historical['result'])
res_btc['time'] = pd.to_datetime(res_btc['time'], unit='ms')


data_ = pd.DataFrame(res_btc.time)

for ticker_ in ['BTC', 'ETH', 'ADA']:
    string_ = 'https://ftx.com/api/markets/' + ticker_ +'-PERP/candles?resolution=300&start_time=' + str(int(start_))            + '&end_time=' + str(int(end_))
    historical = requests.get(string_).json()
    res_ = pd.DataFrame(historical['result'])
    res_['time'] = pd.to_datetime(res_['time'], unit='ms')
    
    data_[ticker_] = res_.close

data_ = data_.resample('60min', on='time').last().drop(['time'], 1)
data_.head(10)



# block where given user-target mean, return the portfolio minimal variance and the (three) weights

datA_ = data_.pct_change().dropna() #  convert to percentage change (lognormal returns)
display(datA_.plot())
display(datA_.mean())

def rnd_weights(n):
    a = np.random.rand(n)
    return a/a.sum()
x = rnd_weights(3) # initialize first guess of weights

# get the boundary values -> maximum and minimum range of returns possible for the portfolio of assets
max_ = max(abs(datA_.mean()))
min_ = min(abs(datA_.mean()))

Ret_ = input('Please enter target return between '+ str(np.round(min_, 6)) + ' and ' + str(np.round(max_, 6)) + ' ')
Ret_ = float(Ret_) # convert target return to float

var = lambda x: x.dot(datA_.cov().dot(x.T)) # function of portfolio variance to be minimized

x_bound = [(-1, 1) for i in range(3)] # we set boundary conditions for weight to be able to be 'short'
x_constraint = ({'type': 'eq', 'fun': lambda x: sum(x) - 1.},               {'type': 'eq', 'fun': lambda x: np.dot(x, datA_.mean()) - Ret_}) # Ret_ is the target return
    
# scipy optimize code
optimal_x = sc.optimize.minimize(var, x, method = 'SLSQP', constraints = x_constraint, bounds = x_bound)

var_ = optimal_x.x.dot(datA_.cov()).dot(optimal_x.x.T)
print('Minimized portfolio variance is', var_, 'and the respective weights are')

dict_output = {'BTC_PERP' : optimal_x.x[0], 'ETH_PERP' : optimal_x.x[1], 'ADA_PERP' : optimal_x.x[2]}
display(dict_output)



# block where we run iterative set of randomized target returns to plot for the efficient returns-variance frontier

mean_ = np.random.uniform(min_, max_, 1000) # set an array of 1,000 runs of target returns within boundary max_, min_
list_ = []

for meaN_ in mean_:
    x = rnd_weights(3) # initialize first guess of weights
    # ret_ = np.dot(x, datA_.mean()) # initialize first guess of weights equivalent returns

    var = lambda x: x.dot(datA_.cov().dot(x.T)) # function of portfolio variance to be minimized

    x_bound = [(-1, 1) for i in range(3)] # we set boundary conditions for weight to be able to be 'short'
    x_constraint = ({'type': 'eq', 'fun': lambda x: sum(x) - 1.},                    {'type': 'eq', 'fun': lambda x: np.dot(x, datA_.mean()) - meaN_}) # meaN_ is the target return
    
    # scipy optimize code
    optimal_x = sc.optimize.minimize(var, x, method = 'SLSQP', constraints = x_constraint, bounds = x_bound)
    vector_ = [meaN_, optimal_x.x.dot(datA_.cov()).dot(optimal_x.x.T), optimal_x.x[0], optimal_x.x[1], optimal_x.x[2]]
    list_.append(vector_)

Res_ = pd.DataFrame(list_, columns = ['mean_', 'variance_', 'wt1', 'wt2', 'wt3'])
print('table of mean-variance optimization with given return, what is variance, and the corresponding weights')
display(Res_.head(5))
plt.scatter(Res_.mean_, Res_.variance_)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(Res_.wt1, Res_.wt2, Res_.wt3)

ax.set_xlabel('weight1')
ax.set_ylabel('weight2')
ax.set_zlabel('weight3')

plt.show()

