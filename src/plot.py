# plot.py
# define functions to plot results and time sequences

import numpy as np
import matplotlib.pyplot as plt

def plot_visits(time_serie):
    days=np.arange(1, time_serie.shape[0]-1)

    plt.figure(figsize=(12, 4))
    plt.plot(days, time_series[1:])
    plt.title(time_series[0])
    plt.ylabel('Views per Page')
    plt.xlabel('Day')
    plt.show()

def plot_loss(history, model_name):
    
    plt.figure(figsize=(12, 4))

    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.title('Loss'.format(model_name))
    plt.ylabel('Views per Page')
    plt.xlabel('Day')
    plt.show()

def plot_pred(train, pred, train_pred, offset): # needs to check if plot is printing on the right day

    train_days = np.arange(1, len(train))
    test_days = np.arange(len(train_pred),  len(train_pred) + len(pred))

    plt.figure(figsize=(12, 4))
    plt.plot(train_days, train[1:len(train)], color='blue')
    plt.plot(test_days+2*(offset+1), pred, color='red')
    plt.plot(train_days[:len(train_pred)] + offset, train_pred, color='green')
    #plt.plot(test_days, train[train_indx:], color='red')
    plt.title(train[0])
    plt.ylabel('Views per Page')
    plt.xlabel('Day')
    
    plt.show()


