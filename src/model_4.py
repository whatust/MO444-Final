# LSTMs
#https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
#https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

#import
print("Importing libraries")
import numpy as np
np.random.seed(0)
from os.path import join
import re
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers import concatenate
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.utils import plot_model
import matplotlib.pyplot as plt
import keras.backend as K

from input_wiki import append_data_lstm
from input_wiki import day_to_value
from input_wiki import date_to_value
from input_wiki import treat

from plot import plot_visits
from plot import plot_loss
from plot import plot_pred

from util import sampe_loss

print("done")

# read data
print("Reading data")
data_path = "../input/train_2.csv"
data_file = open(data_path)
data = data_file.readlines()
days = data[0].split(',')
days = [i.strip().replace("\"", "") for i in days]
days[0] = ""
data = [treat(i) for i in data[1:50001]] # ignores first line and any line after 50000
print("done")

# creates a map where a page is the key and a exclusive integer is the value
m = {}
count = 1
for e in data:
    m[e[0]] = count
    count += 1

# input
print("Creating input")

batch_size = 1000
train_percentage = 0.7
valid_percentage = 0.2

# Creating index for train locaidation and test
train_indx = int(train_percentage * len(data[0]))
valid_indx = int((train_percentage + valid_percentage) * len(data[0]))

train_x = []
train_y = []

valid_x = []
valid_y = []

test_x = []
test_y = []

for i in range(0, len(data)): # using every page
    append_data_lstm(data[i][0], data[i][1:train_indx], m, days, train_x, train_y, batch_size = batch_size)
    append_data_lstm(data[i][0], data[i][train_indx:valid_indx], m, days, valid_x, valid_y, batch_size = batch_size)
    append_data_lstm(data[i][0], data[i][valid_indx:], m, days, test_x, test_y, batch_size = batch_size)

train_x = np.array(train_x).reshape((train_indx - 3, -1, 3))
train_x = np.transpose(train_x, (1,0,2)).reshape(-1, 1, 3)
train_y = np.array(train_y)

valid_x = np.array(valid_x).reshape((valid_indx - train_indx - 2, -1, 3))
valid_x = np.transpose(valid_x, (1,0,2)).reshape(-1, 1, 3)
valid_y = np.array(valid_y)

test_x = np.array(test_x).reshape((len(data[0]) - valid_indx-2, -1, 3))
test_x = np.transpose(test_x, (1,0,2)).reshape(-1, 1, 3)
test_y = np.array(test_y)

print(train_x.shape)
print(valid_x.shape)
print(test_x.shape)

print("done")

# model
print("Initialize model")
epochs = 10
batch_size = 1000
learning_rate = 1e-5
decay = 0.000000
lstm_size = 256
hidden_size = 1024

lstm_input = Input(shape=(1,3), batch_shape=[batch_size, 1 ,3])
lstm_0 = LSTM(lstm_size, return_sequences=True, stateful=True)(lstm_input)
lstm_1 = LSTM(lstm_size, return_sequences=True, stateful=True)(lstm_0)
lstm_2 = LSTM(lstm_size, return_sequences=True, stateful=True)(lstm_1)
lstm_3 = LSTM(lstm_size)(lstm_2)

norm_0 = BatchNormalization()(lstm_3)

dense_input = Input(shape=(3,), batch_shape=[batch_size, 3])
dense_0 = Dense(hidden_size, activation='relu', kernel_regularizer=l2(0.00))(dense_input)
norm_1 = BatchNormalization()(dense_0)

merge = concatenate([norm_1, norm_0], axis=1)

dense_1 = Dense(hidden_size, activation='relu', kernel_regularizer=l2(0.00))(merge)
dense_2 = Dense(hidden_size, activation='relu', kernel_regularizer=l2(0.00))(dense_1)
result = Dense(1, activation='relu')(dense_2)

model = Model(inputs=[lstm_input, dense_input], outputs=result)
opt = Adam(lr=learning_rate, decay=decay)
model.compile(loss=sampe_loss, optimizer=opt)

print("Run model")
history = model.fit([train_x, np.squeeze(train_x, axis=1)], train_y, batch_size=batch_size, epochs=epochs, shuffle=False, verbose=1)
print("done")

t_loss = model.evaluate([train_x, np.squeeze(train_x, axis=1)], y=train_y, batch_size=batch_size)
v_loss = model.evaluate([valid_x, np.squeeze(valid_x, axis=1)], y=valid_y, batch_size=batch_size)
test_loss = model.evaluate([test_x, np.squeeze(test_x, axis=1)], y=test_y, batch_size=batch_size)

print("Model")
print(model)
print("Train Loss:     {}".format(t_loss))
print("Validation Loss:{}".format(v_loss))
print("Test Loss:      {}".format(test_loss))

print("Saving model")
model_name = "model4_{}-{}_{:06.3f}_{:06.3f}.h5".format(lstm_size, hidden_size, v_loss, t_loss)
model_path = join("models", model_name)
#plot_model(model, to_file=join("models", 'model1.png'))
model.save(model_path)
print("Model Saved at: {}".format(model_path))

print("Plot Loss")
plot_loss(history, model_name)

print("Plot Predictions")

pages = []
pages.append(10513) # The big bang theory
pages.append(9033) # Elon Musk
pages.append(10271) # Russia
pages.append(40734) # Thanksgiving

#pages.append(10) # Russia
#pages.append(11) # Russia

train_x = []
train_y = []

valid_x = []
valid_y = []

test_x = []
test_y = []

for i in pages:
    append_data_lstm(data[i][0], data[i][1:train_indx], m, days, train_x, train_y, batch_size = batch_size)
    append_data_lstm(data[i][0], data[i][train_indx:valid_indx], m, days, valid_x, valid_y, batch_size = batch_size)
    append_data_lstm(data[i][0], data[i][valid_indx:], m, days, test_x, test_y, batch_size = batch_size)

train_x = np.expand_dims(np.array(train_x), 1)
train_y = np.array(train_y)

valid_x = np.expand_dims(np.array(valid_x), 1)
valid_y = np.array(valid_y)

test_x = np.expand_dims(np.array(test_x), 1)
test_y = np.array(test_y)

p = model.predict([valid_x, np.squeeze(valid_x, axis=1)], 1).reshape((len(pages), -1))
t = model.predict([train_x, np.squeeze(train_x, axis=1)], 1).reshape((len(pages), -1))

print(p.shape)
print(t.shape)

for idx, page in enumerate(pages):
    plot_pred(data[page][:valid_indx], p[idx], t[idx], batch_size)


