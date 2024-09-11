from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from mh_vae import MHVAE
from mh_utils import buildQ, buildP
import logging as log
from mh_ds import loadDataset


def eval(X, X_hat):
    """Evaluates the reconstruction error.

    Args:
        X (np.ndarray): Batch of images from training set.
        X_hat (np.ndarray): Reconstructed images from VAE.

    Returns:
        float: Mean Squared Error.
    """
    return np.mean(np.square(X - X_hat), axis=(1, 2, 3))

def plotEvalHistogram(model, ds:dict, output_name=None, show=True):
    print('Inside plotEvalHistogram...level 1')
    error = [eval(ds[name], model.predict(ds[name])) for name in ds.keys()]
    plt.figure(figsize=(10, 5))
    print('Inside plotEvalHistogram...level 2')
    [plt.hist(e, bins=50, label=name, density=True, lw=0) for e, name in zip(error, ds.keys())]
    plt.title('Reconstruction Error')
    plt.xlabel('MSE')
    plt.ylabel('Count')
    print('Inside plotEvalHistogram...level 3')
    plt.legend()
    output_name and plt.savefig(output_name) 
    output_name and f'Histogram saved to {output_name}!'
    show and plt.show()
    plt.close()
    print('Inside plotEvalHistogram...level 4')

if __name__ == '__main__':
    log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    ##  Loading data...
    ds_names = ['udacity', 'beamng', 'saevae']
    ds = [loadDataset(ds_name) for ds_name in ds_names]
    ds = map(lambda x: np.concatenate((x[0], x[4]), axis=0), ds)
    ds = {ds_name: ds_ for ds_name, ds_ in zip(ds_names, ds)}
    ##  Building models...
    latent_dim = 200
    model_q = buildQ()
    model_p = buildP()
    ##  MHVAE model...
    model = MHVAE(input_dim=(160, 320, 3), latent_dim=latent_dim, model_p=model_p, model_q=model_q, regularization_const=100000)
    model.compile(optimizer='adam')
    model.load_weights('mh_cvae_weights.h5')
    log.info('\033[92m' + 'Model loaded!' + '\033[0m')
    ##  Evaluate...
    plotEvalHistogram(model, ds, output_name='mh_cvae_eval.pdf')