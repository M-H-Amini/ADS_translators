from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

def buildQ(latent_dim = 20, type_='vae'):
    ###  Q model (encoder)...
    inp = tf.keras.layers.Input(shape=(160, 320, 3))
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation='elu')(inp)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='elu')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), padding='same', activation='elu')(x)
    x = tf.keras.layers.Flatten()(x)
    if type_ == 'vae':
        mean = tf.keras.layers.Dense(latent_dim, activation='tanh')(x)
        log_sigma = tf.keras.layers.Dense(latent_dim)(x)
        model_q = tf.keras.models.Model(inputs=inp, outputs=[mean, log_sigma])
    else:
        z = tf.keras.layers.Dense(latent_dim, activity_regularizer=tf.keras.regularizers.l1(0.01))(x)
        model_q = tf.keras.models.Model(inputs=inp, outputs=z)
    return model_q


def buildP(latent_dim = 20):
    ###  P model (decoder)...
    inp = tf.keras.layers.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(40 * 80 * 32, activation='relu')(inp)
    x = tf.keras.layers.Reshape((40, 80, 32))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='sigmoid')(x)
    model_p = tf.keras.models.Model(inputs=inp, outputs=x)
    return model_p

if __name__ == '__main__':
    model_q = buildQ()
    model_q.summary()
    model_p = buildP()
    model_p.summary()