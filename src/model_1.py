# First Model using 2 fully connected

#import
print("Importing libraries")
import numpy as np
np.random.seed(0)
from os.path import join
import re
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
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
data = [treat(i) for i in data[1:50001]] # ignores first line and any line after 50000
print("done")

# creates a map where a page is the key and a exclusive integer is the value
m = {}
count = 1
for e in data:
    m[e[0]] = count
    count += 1

train_percentage = 0.7
valid_percentage = 0.2

#Separating train validation test
train_indx = int(train_percentage * len(data[0]))
valid_indx = int((train_percentage + valid_percentage) * len(data[0]))

# input
print("Creating input")

train_x = []
train_y = []

valid_x = []
valid_y = []

test_x = []
test_y = []

for i in range(0, len(data)): # using every page
    append_data(data[i][0], data[i][1:train_indx], m, days, train_x, train_y, num_days = 7)
    append_data(data[i][0], data[i][train_indx:valid_indx], m, days, valid_x, valid_y, num_days = 7)
    append_data(data[i][0], data[i][valid_indx:], m, days, test_x, test_y, num_days = 7)

train_x = np.array(train_x)
train_y = np.array(train_y)

valid_x = np.array(valid_x)
valid_y = np.array(valid_y)

test_x = np.array(test_x)
test_y = np.array(test_y)

print("done")

# model
print("Initialize model")
epochs = 20
batch_size = 1000
learning_rate = 1e-7
decay = 0.000000
hidden_size = 1024 

model = Sequential()

model.add(Dense(hidden_size, input_shape=train_x[0].shape, activation='relu', kernel_regularizer=l2(0.00)))
model.add(Dense(hidden_size, activation='relu', kernel_regularizer=l2(0.00)))
model.add(Dense(hidden_size, activation='relu', kernel_regularizer=l2(0.00)))
model.add(Dense(1, activation='relu'))

opt = Adam(lr=learning_rate, decay=decay)
model.compile(loss=sampe_loss, optimizer=opt)

print("Run model")
history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1)
print("done")

t_loss = model.evaluate(x=train_x, y=train_y)
v_loss = model.evaluate(x=valid_x, y=valid_y)
test_loss = model.evaluate(x=test_x, y=test_y)

print("Model")
print(model)
print("Train Loss:     {}".format(t_loss))
print("Validation Loss:{}".format(v_loss))
print("Test Loss:      {}".format(test_loss))

print("Saving model")
model_name = "model1_{}_{:06.3f}_{:06.3f}.h5".format(hidden_size, v_loss, t_loss)
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
pages.append(40734) # Thanksgiving
pages.append(10271) # Russia

train_x = []
train_y = []

valid_x = []
valid_y = []

test_x = []
test_y = []

for i in pages:
    append_data(data[i][0], data[i][1:train_indx], m, days, train_x, train_y, num_days = 7)
    append_data(data[i][0], data[i][train_indx:valid_indx], m, days, valid_x, valid_y, num_days = 7)
    append_data(data[i][0], data[i][valid_indx:], m, days, test_x, test_y, num_days = 7)

train_x = np.array(train_x)
train_y = np.array(train_y)

valid_x = np.array(valid_x)
valid_y = np.array(valid_y)

test_x = np.array(test_x)
test_y = np.array(test_y)

p = model.predict(valid_x, 1).reshape((len(pages), -1))
t = model.predict(train_x, 1).reshape((len(pages), -1))
ts = model.predict(test_x, 1).reshape((len(pages), -1))

print(p.shape)
print(t.shape)

for idx, page in enumerate(pages):
    plot_pred(data[page], p[idx], t[idx], ts[idx], 7)


