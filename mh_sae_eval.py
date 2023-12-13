from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from mh_sae import MHAE
from mh_dave2_data import prepareDataset as prepareDatasetUdacity
from mh_utils import buildQ, buildP
import logging as log
from mh_ds import loadDataset

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##  Loading data...
ds_names = ['udacity', 'beamng']#, 'cats_vs_dogs']
ds = [loadDataset(ds_name) for ds_name in ds_names]
ds = map(lambda x: np.concatenate((x[0], x[4]), axis=0), ds)  ##  Concatenate train and test sets
ds = {ds_name: ds_ for ds_name, ds_ in zip(ds_names, ds)}

##  Building models...
latent_dim = 20

model_q = buildQ(type_='sae')
model_p = buildP()

###  MHSAE model...
model = MHAE(input_dim=(66, 200, 3), latent_dim=latent_dim, model_p=model_p, model_q=model_q)
model.compile(optimizer='adam')
model.load_weights('mh_csae_weights.h5')
log.info('\033[92m' + 'Model loaded!' + '\033[0m')


def eval(X, X_hat):
    """Evaluates the reconstruction error.

    Args:
        X (np.ndarray): Batch of images from training set.
        X_hat (np.ndarray): Reconstructed images from VAE.

    Returns:
        float: Mean Squared Error.
    """
    return np.mean(np.square(X - X_hat), axis=(1, 2, 3))

error = [eval(ds[name], model.predict(ds[name])) for name in ds_names]

plt.figure(figsize=(10, 5))

plt.legend()
##  Plotting histograms with x axis in log scale...
[plt.hist(e, bins=50, label=name, density=True) for e, name in zip(error, ds_names)]
# plt.gca().set_xscale('log')
plt.title('Reconstruction Error')
plt.xlabel('MSE')
plt.ylabel('Count')
plt.legend()
plt.savefig('mh_csae_eval.pdf')
plt.show()