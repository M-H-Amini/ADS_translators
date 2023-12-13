import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mh_sae import MHAE
from mh_vae import MHVAE
from mh_utils import buildQ, buildP
from mh_ds import loadDataset
from mh_ds import visualize as vis
import logging as log
from mh_vae_eval import plotEvalHistogram
import os
from PIL import Image
from tqdm import tqdm

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##  Make sure tensorflow is using GPU...
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    log.info(f'GPUs: {gpus}')
    try:
        # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
        log.info('\033[92m' + 'GPU found!' + '\033[0m')
    except RuntimeError as e:
        log.error(e)

#  Loading data...
# X_train, y_train, X_val, y_val, X_test, y_test = loadDataset('beamng')

train = not True

##  VAE model...
latent_dim = 20
model_q_vae = buildQ()
model_p_vae = buildP()
model_q_vae.trainable = False
model_p_vae.trainable = False
model_vae = MHVAE(input_dim=(160, 320, 3), latent_dim=latent_dim, model_p=model_p_vae, model_q=model_q_vae, regularization_const=100000)
model_vae.trainable = False
model_vae.compile(optimizer='adam')
model_vae.load_weights('mh_cvae_weights.h5')
log.info('\033[92m' + 'VAE model loaded!' + '\033[0m')

##  SAE...
model_name = 'mh_csae_cvae_beamng'
latent_dim = 20
model_q_sae = buildQ(type_='sae')
model_q_sae.summary()
model_p_sae = buildP()
model_p_sae.summary()
model_sae = MHAE(input_dim=(160, 320, 3), latent_dim=latent_dim, model_p=model_p_sae, model_q=model_q_sae)
model_sae.compile(optimizer='adam')
model_sae.load_weights(f'mh_sae_beamng_weights.h5')
log.info('\033[92m' + 'SAE model loaded!' + '\033[0m')

model_sae.load_weights(f'{model_name}_weights.h5')
img = plt.imread('beamng_sample.png').astype(np.float32) / 255.
output_folder = os.getcwd()
os.makedirs(output_folder, exist_ok=True)
X = np.array([img])
X_hat = model_sae.predict(X)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(X[0])
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(X_hat[0])
plt.title('Reconstructed')
plt.savefig(f'{output_folder}/beamng_reconstruction.png')
plt.show()
