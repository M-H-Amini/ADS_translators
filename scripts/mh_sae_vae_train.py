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
import sys 

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

##  Loading data...
dataset = sys.argv[1] if len(sys.argv) > 1 else 'mnist'  ##  It should be `beamng`. Use `mnist` for a simple showcase...
X_train, y_train, X_val, y_val, X_test, y_test = loadDataset(dataset)

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
##  Training...
N = len(X_train)
batch_size = 32
losses = []

def trainEpoch(indices):
    for i in range(N // batch_size):
        with tf.GradientTape(persistent=True) as g:
            g.watch(model_sae.trainable_variables)
            X = X_train[indices[i*batch_size:(i+1)*batch_size]]
            X_hat = model_sae(X)
            X_hat_hat = model_vae(X_hat)
            loss_sae = tf.reduce_mean(tf.square(X - X_hat))
            loss_vae = tf.reduce_mean(tf.square(X_hat - X_hat_hat))
            loss = loss_sae + 0.5 * loss_vae
        grads = g.gradient(loss, model_sae.trainable_variables)
        model_sae.optimizer.apply_gradients(zip(grads, model_sae.trainable_variables))
        losses.append(loss.numpy())
        log.info(f'Batch {i+1}/{N//batch_size} - Loss: {loss.numpy()}')
    return loss.numpy()

def visualize(X, X_hat):
    print('Visualizing...')
    h, w = 160, 320
    indices = np.random.permutation(len(X))
    img = np.zeros((h*4, w*8, 3))
    for i in range(4):
        for j in range(4):
            img[i*h:(i+1)*h, 2*j*w:(2*j+1)*w, :] = X[indices[i*4+j]]
            img[i*h:(i+1)*h, (2*j+1)*w:(2*j+2)*w, :] = X_hat[indices[i*4+j]]
    plt.imshow(img)
    plt.show(block=False)
    plt.pause(0.1)
    frames.append(img)
    print('Visualized!')
    # plt.close()



ds_names = ['udacity', 'beamng']
ds = [loadDataset(ds_name) for ds_name in ds_names]
ds = map(lambda x: np.concatenate((x[0], x[4]), axis=0), ds)
ds = {ds_name: ds_ for ds_name, ds_ in zip(ds_names, ds)}
ds['transformed_beamng'] = ds['beamng']
best_loss = 1000

if train:
    plt.figure(figsize=(16, 8))
    frames = []
    plotEvalHistogram(model_vae, ds, output_name=f'mh_csae_cvae_eval_{0}.pdf', show=False)
    visualize(X_train, model_sae.predict(X_train))

    for epoch in range(10):
        log.info(f'Epoch {epoch+1}/{10}')
        indices = np.random.permutation(N)
        loss = trainEpoch(indices)
        ##  Plotting...
        X_hat = model_sae.predict(X_train)
        ds['transformed_beamng'] = X_hat
        plotEvalHistogram(model_vae, ds, output_name=f'mh_csae_cvae_eval_{epoch}.pdf', show=False)
        visualize(X_train, X_hat)
        if loss < best_loss:
            best_loss = loss
            print('New best loss!', best_loss)
            model_sae.save_weights(f'{model_name}_weights.h5')



    plt.figure()
    plt.plot(losses)
    plt.title('Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()
    

    ##  Generate GIF...
    import imageio
    imageio.mimsave(f'{model_name}.gif', frames, fps=5)


model_sae.load_weights(f'{model_name}_weights.h5')
output_folder = 'ds_beamng_saevae'
os.makedirs(output_folder, exist_ok=True)
X = np.concatenate((X_train, X_val, X_test), axis=0)
y = np.concatenate((y_train, y_val, y_test), axis=0)
X_hat = model_sae.predict(X)
df_dict = {'img': [], 'steer': []}
for i in tqdm(range(len(X_hat))):
    img = (X_hat[i] * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img_name = f'img_{i}.jpg'
    img.save(os.path.join(output_folder, img_name))
    df_dict['img'].append(img_name)
    df_dict['steer'].append(y[i])
df = pd.DataFrame(df_dict)
df.to_csv(os.path.join(output_folder, 'labels.csv'), index=False)
vis(X, X_hat, 'saevae.pdf')
log.info('\033[92m' + 'Dataset generated!' + '\033[0m')
