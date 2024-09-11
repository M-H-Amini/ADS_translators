from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from mh_vae import MHVAE
from mh_ds import loadDataset
from mh_utils import buildQ, buildP
import logging as log
import sys

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##  Loading data...
dataset = sys.argv[1] if len(sys.argv) > 1 else 'mnist'  #
input_shape = (160, 320, 3) 
# dataset = 'kittt'  ##  It should be `kitti`. Use `mnist` for a simple showcase...
X_train, _, X_val, _, X_test, _ = loadDataset(dataset, 1000)
print(input_shape)

##  Building models...
latent_dim = 20

model_q = buildQ()
model_q.summary()

model_p = buildP()
model_p.summary()

###  MHVAE model...
model = MHVAE(input_dim=input_shape, latent_dim=latent_dim, model_p=model_p, model_q=model_q, regularization_const=100000, train_visualize=True)
model.compile(optimizer='adam', run_eagerly=True)
# model.load_weights('mh_cvae_kitti_weights.h5')
log.info('\033[92m' + 'Model loaded!' + '\033[0m')
model.fit(X_train, epochs=20, batch_size=64)
# model.generateGIF(f'mh_cvae_{dataset}.gif')
model.save_weights(f'mh_cvae_{dataset}_weights.h5')