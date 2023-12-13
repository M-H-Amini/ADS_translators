from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from mh_vae import MHVAE
from mh_ds import loadDataset
from mh_utils import buildQ, buildP
import logging as log

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##  Loading data...
dataset = 'kittt'
X_train, X_val, X_test = loadDataset(dataset, 1000)

##  Building models...
latent_dim = 20

model_q = buildQ()
model_q.summary()

model_p = buildP()
model_p.summary()

###  MHVAE model...
model = MHVAE(input_dim=(160, 320, 3), latent_dim=latent_dim, model_p=model_p, model_q=model_q, regularization_const=100000, train_visualize=True)
model.compile(optimizer='adam', run_eagerly=True)
model.load_weights('mh_cvae_kitti_weights.h5')
log.info('\033[92m' + 'Model loaded!' + '\033[0m')
model.fit(X_train, epochs=20, batch_size=64)
model.generateGIF(f'mh_cvae_{dataset}.gif')
model.save_weights(f'mh_cvae_{dataset}_weights.h5')