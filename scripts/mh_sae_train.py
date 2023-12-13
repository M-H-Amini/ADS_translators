import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from mh_sae import MHAE
from mh_utils import buildQ, buildP
from mh_ds import loadDataset
import logging as log

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##  Make sure tensorflow is using GPU...
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    log.info(f'GPUs: {gpus}')
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
    except RuntimeError as e:
        log.error(e)

##  Loading data...
X_train_u, y_train_u, X_val_u, y_val_u, X_test_u, y_test_u = loadDataset('beamng')

##  Building models...
model_name = 'mh_sae_beamng'
latent_dim = 20

model_q = buildQ(type_='sae')
model_q.summary()

model_p = buildP()
model_p.summary()

###  MHVAE model...
model = MHAE(input_dim=(160, 320, 3), latent_dim=latent_dim, model_p=model_p, model_q=model_q, train_visualize=True)
model.compile(optimizer='adam', run_eagerly=True)
# model.load_weights(f'{model_name}_weights.h5') if os.path.exists(f'{model_name}_weights.h5') else None
log.info('\033[92m' + 'Model loaded!' + '\033[0m')
model.fit(X_train_u, epochs=20, batch_size=32)
model.generateGIF(f'{model_name}.gif')
model.save_weights(f'{model_name}_weights.h5')