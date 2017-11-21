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
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
print("done")

def append_data(dat, m, days, x, y, num_days = 7):
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

# read data
print("Reading data")
data_path = "train_1.csv"
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
for i in range(0, 1): # using only the first page
# for i in range(0, len(data)) # using every page
    append_data(data[i], m, days, x, y)
x = np.array(x)
y = np.array(y)
print("done")

# model
print("Initialize model")
epochs = 2000
batch_size = 64
learning_rate = 1e-6
decay = 0.00
model = Sequential()
model.add(Dense(512, input_shape=x[0].shape, activation='relu'))
model.add(Dense(512, activation='relu'))

model.add(Dense(1, activation='relu'))

opt = Adam(lr=learning_rate, decay=decay)
model.compile(loss='mean_squared_error',
              optimizer=opt)

print("Run model")
model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=1)
print("done")

## Check result for a given page and day
x = []
y = []
append_data(data[0], m, days, x, y)
x = np.array([x[0]])
y = np.array([y[0]])

print(model.predict(x, 1, verbose=1))
print(y)
