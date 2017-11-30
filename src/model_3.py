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
from keras.layers import GRU
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

from input_wiki import append_data 
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

num_days = 1
train_percentage = 0.7
valid_percentage = 0.2

# Creating index for train lcaidation and test
train_indx = int(train_percentage * len(data[0]))
valid_indx = int((train_percentage + valid_percentage) * len(data[0]))

train_x = []
train_y = []

valid_x = []
valid_y = []

test_x = []
test_y = []

for i in range(0, len(data)): # using every page
    append_data(data[i][0], data[i][1:train_indx], m, days, train_x, train_y, num_days = num_days)
    append_data(data[i][0], data[i][train_indx:valid_indx], m, days, valid_x, valid_y, num_days = num_days)
    append_data(data[i][0], data[i][valid_indx:], m, days, test_x, test_y, num_days = num_days)

train_x = np.expand_dims(np.array(train_x), 1)
train_y = np.array(train_y)

valid_x = np.expand_dims(np.array(valid_x), 1)
valid_y = np.array(valid_y)

test_x = np.expand_dims(np.array(test_x), 1)
test_y = np.array(test_y)

print("done")

# model
print("Initialize model")
epochs = 5
batch_size = 1000
learning_rate = 1e-5
decay = 0.000000
lstm_size = 256
hidden_size = 1024

lstm_input = Input(shape=(1,3))
lstm_0 = GRU(lstm_size, return_sequences=True)(lstm_input)
lstm_1 = GRU(lstm_size, return_sequences=True)(lstm_0)
lstm_2 = GRU(lstm_size, return_sequences=True)(lstm_1)
lstm_3 = GRU(lstm_size)(lstm_2)

norm_0 = BatchNormalization()(lstm_3)

dense_input = Input(shape=(3, ))
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
history = model.fit([train_x, np.squeeze(train_x, axis=1)], train_y, batch_size=batch_size, epochs=epochs, verbose=1)
print("done")

t_loss = model.evaluate([train_x, np.squeeze(train_x, axis=1)], y=train_y)
v_loss = model.evaluate([valid_x, np.squeeze(valid_x, axis=1)], y=valid_y)

print("Model")
print(model)
print("Train Loss:     {}".format(t_loss))
print("Validation Loss:{}".format(v_loss))

print("Saving model")
model_name = "model3_{}-{}_{:06.3f}_{:06.3f}.h5".format(lstm_size, hidden_size, v_loss, t_loss)
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
    append_data(data[i][0], data[i][1:train_indx], m, days, train_x, train_y, num_days = num_days)
    append_data(data[i][0], data[i][train_indx:valid_indx], m, days, valid_x, valid_y, num_days = num_days)
    append_data(data[i][0], data[i][valid_indx:], m, days, test_x, test_y, num_days = num_days)

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
    plot_pred(data[page][:valid_indx], p[idx], t[idx], num_days)


