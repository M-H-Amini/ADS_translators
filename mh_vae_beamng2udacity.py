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
X_train, y_train, X_val, y_val, X_test, y_test = loadDataset('beamng')
X = np.concatenate((X_train, X_val, X_test), axis=0)
X = X[:100]

##  Building models...
latent_dim = 20

model_q = buildQ()
model_q.summary()

model_p = buildP()
model_p.summary()

###  MHVAE model...
model = MHVAE(input_dim=(66, 200, 3), latent_dim=latent_dim, model_p=model_p, model_q=model_q, regularization_const=100000)
model.compile(optimizer='adam')
model.load_weights('mh_cvae_weights.h5')
log.info('\033[92m' + 'Model loaded!' + '\033[0m')
pred = model.predict(X)

fig, axs = plt.subplots(len(X), 2, figsize=(10, 1.5 * len(X)))
for i in range(len(X)):
    axs[i, 0].imshow(X[i])
    axs[i, 1].imshow(pred[i])
    axs[i, 0].axis('off')
    axs[i, 1].axis('off')

plt.savefig('mh_cvae_beamng2udacity.pdf')
plt.close()