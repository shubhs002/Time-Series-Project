
# coding: utf-8

# In[17]:


# import required packages

import numpy as np # linear algebra
from numpy import newaxis
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
import seaborn as sb
sb.set_style('whitegrid')

import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima_model import ARIMA

import sklearn
import sklearn.preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import  train_test_split
from sklearn.preprocessing import MinMaxScaler

from subprocess import check_output
import time #helper libraries

import math

import datetime

import os

import tensorflow as tf
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.rnn as rnn

import random


# In[2]:


#Import S&P 500 file and keep only 'Close' values. Drop other columns
snp = pd.read_csv('HistoricalPrices.csv')
print (snp.head())
snp.drop(['Open', 'High', 'Low'], axis=1,inplace=True)
snp.head()


# In[3]:


# Import gdp data
gdp = pd.read_csv('GDP Historical Data.csv')
print (gdp.head())
gdp.head()


# In[4]:


#Convert date to Index
snp.Date = pd.to_datetime(snp.Date)
snp.set_index('Date', inplace=True)
snp.head()


# In[5]:


#Convert date to Index
gdp.Date = pd.to_datetime(gdp.Date)
gdp.set_index('Date', inplace=True)
gdp.head()


# In[6]:


# Set in Increasing Order
snp.sort_index(ascending=True, inplace=True)
snp.head()


# In[7]:


# Set in Increasing Order
gdp.sort_index(ascending=True, inplace=True)
gdp.head()


# In[8]:


# Plot S&P Data
snp.plot(figsize=(20,10), linewidth=2, fontsize=20)
plt.xlabel('Year', fontsize=20);
plt.axvspan('2001-03-01', '2001-11-30', facecolor='0.8', alpha = 0.5)
plt.axvspan('2007-12-01', '2009-06-30', facecolor='0.8', alpha = 0.5)
plt.savefig('snp_daily.jpeg',format ='jpeg')


# In[9]:


# Convert S&P daily  to S&P Weekly
Close = snp.Close.resample('W-MON', how='last')
weekly_data = pd.DataFrame(Close)
weekly_data.head()


# In[10]:


# Plot Weekly data
weekly_data.plot(figsize=(20,10), linewidth=2, fontsize=20)
plt.xlabel('Year', fontsize=20);
plt.axvspan('2001-03-01', '2001-11-30', facecolor='0.8', alpha = 0.5)
plt.axvspan('2007-12-01', '2009-06-30', facecolor='0.8', alpha = 0.5)


# In[11]:


# Rolling average of weekly data using 12 weeks
Price = weekly_data[['Close']]
Price.rolling(12).mean().plot(figsize=(20,10), linewidth=2, fontsize=20)
plt.axvspan('2001-03-01', '2001-11-30', facecolor='0.8', alpha = 0.5)
plt.axvspan('2007-12-01', '2009-06-30', facecolor='0.8', alpha = 0.5)
plt.xlabel('Year', fontsize=20);


# In[10]:


# First order difference of weekly data
Order1_Price = Price.diff()
Order1_Price.plot(figsize=(20,10), linewidth=2, fontsize=20)
plt.axvspan('2001-03-01', '2001-11-30', facecolor='0.8', alpha = 0.5)
plt.axvspan('2007-12-01', '2009-06-30', facecolor='0.8', alpha = 0.5)
plt.xlabel('Year', fontsize=20);


# In[35]:


# Auto correlation of weekly data
pd.plotting.autocorrelation_plot(Price);
plt.savefig('AutoCorrelation.jpeg',format ='jpeg')


# In[12]:


# Convert s&p daily to quarterly
Close = snp.Close.resample('Q', how='last')
quarterly_data = pd.DataFrame(Close)
quarterly_data.head()


# In[13]:


# Normalise S&P quarterly and gdp  to fall in to same scale and plot them.
concat = pd.concat([quarterly_data,gdp],axis=1)
from sklearn import preprocessing

#x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(concat.iloc[1:-4,:])
df = pd.DataFrame(x_scaled)
df = df.set_index(concat.index.values[1:-4])


df.plot(figsize=(20,10), linewidth=2, fontsize=20)
plt.xlabel('Year', fontsize=20);
plt.axvspan('2001-03-31', '2001-11-30', facecolor='0.8', alpha = 0.5)
plt.axvspan('2007-12-31', '2009-06-30', facecolor='0.8', alpha = 0.5)
plt.legend(['S&P 500','GDP'], fontsize=18)

plt.savefig('snp_qtrly vs gdp.jpeg',format ='jpeg')


# In[14]:


# Rolling average of s&p quarterly using 4 quarters
Price = quarterly_data[['Close']]
Price.rolling(4).mean().plot(figsize=(20,10), linewidth=2, fontsize=20)
plt.axvspan('2001-03-01', '2001-11-30', facecolor='0.8', alpha = 0.5)
plt.axvspan('2007-12-01', '2009-06-30', facecolor='0.8', alpha = 0.5)
plt.xlabel('Year', fontsize=20);

plt.savefig('snp_trend.jpeg',format ='jpeg')


# In[15]:


# 1st order difference plot
Order1_Price = Price.diff()
Order1_Price.plot(figsize=(20,10), linewidth=2, fontsize=20)
plt.axvspan('2001-03-01', '2001-11-30', facecolor='0.8', alpha = 0.5)
plt.axvspan('2007-12-01', '2009-06-30', facecolor='0.8', alpha = 0.5)
plt.xlabel('Year', fontsize=20);
plt.savefig('Stationarity.jpeg',format ='jpeg')


# In[18]:


# Auto-correlation of 1st oredr differnce
pd.plotting.autocorrelation_plot(Order1_Price[1:]);
plt.savefig('AutoCorr_Order1.jpeg',format ='jpeg')


# In[19]:


# data prep for RNN
test_set_size_percentage = 20
# function to create train, test data given stock data and sequence length
def load_data(stock, seq_len):
    data_raw = stock.as_matrix() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len): 
        data.append(data_raw[index: index + seq_len])
    
    data = np.array(data);
    test_set_size = int(np.round(test_set_size_percentage/100*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    
    x_test = data[train_set_size:,:-1,:]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]


# In[20]:


seq_len = 20
x_train, y_train, x_test, y_test = load_data(quarterly_data,seq_len)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ',y_test.shape)


# In[21]:


## Basic Cell RNN in tensorflow

index_in_epoch = 0;
perm_array  = np.arange(x_train.shape[0])
np.random.shuffle(perm_array)

# function to get the next batch
def get_next_batch(batch_size):
    global index_in_epoch, x_train, perm_array   
    start = index_in_epoch
    index_in_epoch += batch_size
    
    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array) # shuffle permutation array
        start = 0 # start next epoch
        index_in_epoch = batch_size
        
    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]


# In[22]:


# parameters
n_steps = seq_len-1 
n_inputs = 1
n_neurons = 500
n_outputs = 1
n_layers = 1
learning_rate = 0.001
batch_size = 20
n_epochs = 200
train_set_size = x_train.shape[0]
test_set_size = x_test.shape[0]

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

# use Basic RNN Cell
layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
         for layer in range(n_layers)]

#use Basic LSTM Cell 
#layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.elu)
#          for layer in range(n_layers)]

# use LSTM Cell with peephole connections
#layers = [tf.contrib.rnn.LSTMCell(num_units=n_neurons, 
#                                  activation=tf.nn.leaky_relu, use_peepholes = True)
#          for layer in range(n_layers)]

# use GRU cell
#layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.leaky_relu)
#          for layer in range(n_layers)]
                                                                     
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons]) 
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
outputs = outputs[:,n_steps-1,:] # keep only last output of sequence
                                              
loss = tf.reduce_mean(tf.square(outputs - y)) # loss function = mean squared error 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
training_op = optimizer.minimize(loss)
                                              
# run graph
with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    for iteration in range(int(n_epochs*train_set_size/batch_size)):
        x_batch, y_batch = get_next_batch(batch_size) # fetch the next training batch 
        sess.run(training_op, feed_dict={X: x_batch, y: y_batch}) 
        if iteration % int(5*train_set_size/batch_size) == 0:
            mse_train = loss.eval(feed_dict={X: x_train, y: y_train}) 
            print('%.2f epochs: MSE train = %.6f'%(iteration*batch_size/train_set_size, mse_train))

    y_train_pred = sess.run(outputs, feed_dict={X: x_train})
    y_test_pred = sess.run(outputs, feed_dict={X: x_test})


# In[23]:


ft = 0 # 0 = open, 1 = close, 2 = highest, 3 = lowest

## show predictions
#plt.figure(figsize=(15, 5));
#plt.subplot(1,2,1);
rcParams['figure.figsize'] = 20, 10

plt.plot(np.arange(y_train.shape[0]), y_train[:,ft], color='blue', label='train target')


plt.plot(np.arange(y_train.shape[0],
                   y_train.shape[0]+y_test.shape[0]),
         y_test[:,ft], color='black', label='test target')

plt.plot(np.arange(y_train_pred.shape[0]),y_train_pred[:,ft], color='red',
         label='train prediction')


plt.plot(np.arange(y_train_pred.shape[0],
                   y_train_pred.shape[0]+y_test_pred.shape[0]),
         y_test_pred[:,ft], color='green', label='test prediction')

plt.title('past and future stock prices', fontsize=18)
plt.xlabel('time [days]', fontsize=18)
plt.ylabel('normalized price', fontsize=18)
plt.legend(loc='best');
plt.savefig('past&future.jpeg',format ='jpeg')
plt.show()



#plt.subplot(1,2,2);

plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_test.shape[0]),
         y_test[:,ft], color='black', label='test target')

plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_test_pred.shape[0]),
         y_test_pred[:,ft], color='green', label='test prediction')


plt.title('future stock prices', fontsize=18)
plt.xlabel('time [days]', fontsize=18)
plt.ylabel('normalized price', fontsize=18)
plt.legend(loc='best', fontsize=18);
plt.savefig('future.jpeg',format ='jpeg')
plt.show()


corr_price_development_train = np.sum(np.equal(np.sign(-y_train[:,0]),
            np.sign(y_train_pred[:,0])).astype(int)) / y_train.shape[0]

corr_price_development_test = np.sum(np.equal(np.sign(y_test[:,0]),
            np.sign(y_test_pred[:,0])).astype(int)) / y_test.shape[0]

print('correct sign prediction for close price for train/test: %.2f/%.2f'%(
    corr_price_development_train, corr_price_development_test))



