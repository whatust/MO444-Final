# util.py
# Defiene misc functions to work on the model

import keras.backend as K

def sampe_loss(y_p, y_t):

    return 200 * K.mean(K.abs((y_t - y_p) / K.clip(K.abs(y_t) + K.abs(y_p), K.epsilon(), None)))

