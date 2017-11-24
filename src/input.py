# LSTMs
#https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
#https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

#import
print("Importing libraries")
import numpy as np
np.random.seed(0)
import re
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import keras.backend as K
print("done")

def append_data(dat, m, days, x, y, num_days = 1):
    ### Appends to x, y the input/output
    ### input format: [page, current_day, access_of_previous_days]
    # page: a exclusive number given by the map m
    # current_day: a integer for the given month and days
    # access_of_previous_days: array of access of *num_days* previous days
    ### output: correct_number_of_access
    for i in range(num_days + 1, len(dat)):
        x += [[m[dat[0]]]]
        x[-1] += [date_to_value(days[i])]
        for j in range(1, num_days + 1):
            x[-1] += [dat[i - j]]
        y += [dat[i]]
    return x, y

def day_to_value(month, day):
    ## Given a given month and day, returns a fixed integer
    return month * 31 + day

def date_to_value(st):
    ## Given a string YYYY-MM-DD, returns a fixed integer (ignores the year)
    st = st.split('-')
    return day_to_value(int(st[1]), int(st[2]))

def treat(s):
    ## convert a line of input to an int array
    r = [""]
    f = False
    for i in range(0, len(s)):
        if(s[i] == "\""):
            f = not f
        if(f):
            r[-1] += s[i]
        else:
            if(s[i] == ','):
                r += [""]
            else:
                r[-1] += s[i]
    r = [i.strip() for i in r]
    for i in range(1, len(r)):
        if(r[i] == ""):
            r[i] = 0
        else:
            r[i] = int(float(r[i]))
    return r

def sampe_loss(y_p, y_t):

    return 200 * K.mean(K.abs((y_t - y_p) / K.clip(K.abs(y_t) + K.abs(y_p), K.epsilon(), None)))

def plot_visits(index):
    days=np.arange(1, select_train.shape[1])

    plt.figure(figsize=(12, 4))
    plt.plot(days, select_train.loc[index][1:])
    plt.title(select_train.loc[index]['Page'])
    plt.ylabel('Views per Page')
    plt.xlabel('Day')
    plt.show()

def plot_loss(history):
    
    plt.figure(figsize=(12, 4))

    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.title('Loss')
    plt.ylabel('Views per Page')
    plt.xlabel('Day')
    plt.show()

def plot_pred(train, pred, train_pred): # needs to check if plot is printing on the right day
    train_days = np.arange(1, valid_indx)
    test_days = np.arange(train_indx + 1, valid_indx + 1)

    plt.figure(figsize=(12, 4))
    plt.plot(train_days, train[1:valid_indx], color='blue')
    plt.plot(test_days, pred, color='red')
    plt.plot(train_days[:len(train_pred)], train_pred, color='green')
    #plt.plot(test_days, train[train_indx:], color='red')
    plt.title(train[0])
    plt.ylabel('Views per Page')
    plt.xlabel('Day')
    
    plt.show()

# read data
print("Reading data")
data_path = "../input/train_1.csv"
data_file = open(data_path)
data = data_file.readlines()
days = data[0].split(',')
days = [i.strip().replace("\"", "") for i in days]
days[0] = ""
data = [treat(i) for i in data[1:50000]] # ignores first line and any line after 50000
print("done")

# creates a map where a page is the key and a exclusive integer is the value
m = {}
count = 1
for e in data:
    m[e[0]] = count
    count += 1

# input
print("Creating input")
x = []
y = []

#pages = 10513 # The big bang theory
#pages = 9033 # Elon Musk
pages = 40734 # Thanksgiving
#pages = 10271 # Russia

for i in [pages]: # using only the first page
#for i in range(0, 10): # using every page
    append_data(data[i], m, days, x, y)
x = np.expand_dims(np.array(x), axis=1)
y = np.array(y)


train_percentage = 0.7
valid_percentage = 0.2

#Separating train validation test
train_indx = int(train_percentage * x.shape[0])
valid_indx = int((train_percentage + valid_percentage) * x.shape[0])

train_x = x[:train_indx]
train_y = y[:train_indx]

valid_x = x[train_indx:valid_indx]
valid_y = y[train_indx:valid_indx]

test_x = x[valid_indx:]
test_y = y[valid_indx:]

print("done")

# model
print("Initialize model")
epochs = 1000
batch_size = 10
learning_rate = 1e-4
decay = 0.00
model = Sequential()
model.add(LSTM(150, input_shape=(1,3), return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(150, return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(150, return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(150, return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(150))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.00)))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.00)))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.00)))
#model.add(BatchNormalization())
#model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
#model.add(BatchNormalization())
#model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))

model.add(Dense(1, activation='relu'))

opt = Adam(lr=learning_rate, decay=decay)
model.compile(loss=sampe_loss, optimizer=opt)

print("Run model")
history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1)
print("done")

plot_loss(history)

p = np.squeeze(model.predict(valid_x, 1))
t = np.squeeze(model.predict(train_x, 1))
plot_pred(data[pages], p, t)

print("Train Loss:     {}".format(model.evaluate(x=train_x, y=train_y)))
print("Validation Loss:{}".format(model.evaluate(x=valid_x, y=valid_y)))

