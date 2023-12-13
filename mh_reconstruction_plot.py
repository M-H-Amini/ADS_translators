import os 
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from mh_sae import MHAE
from mh_vae import MHVAE
from mh_utils import buildQ, buildP
import logging as log
from PIL import Image
from mh_stat_test import statisticalTests

input_folders = ['ds_udacity/images', 'ds_beamng/images', 'ds_beamng_saevae', 'ds_beamng_cycle/images', 'ds_beamng_style']
names = 'DS_real, DS_sim, SAEVAE, cycleG, styleT'.split(', ')
##  Read all jpg or png files in the input folder...
img_files = {name: glob(os.path.join(ds, '*.jpg')) + glob(os.path.join(ds, '*.png')) for ds, name in zip(input_folders, names)}
real_len = len(img_files['DS_real'])
img_files['DS_real,train'] = img_files['DS_real'][:int(0.7 * real_len)]
img_files['DS_real,test'] = img_files['DS_real'][int(0.7 * real_len):]
del img_files['DS_real']
names = 'DS_real,train, DS_real,test, DS_sim, SAEVAE, cycleG, styleT'.split(', ')

min_length = min([len(img_files[name]) for name in names])
length = 2000
img_files = {name: img_files[name][:length] for name in names}

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

##  Reconstruction...
errors = {name: [] for name in names}

for name in names:
    print(f'Processing {name}...')
    X = []
    for img_file in tqdm(img_files[name]):
        img = Image.open(img_file).resize((320, 160)).convert('RGB')
        img = np.array(img) / 255.
        X.append(img)
    X = np.array(X)
    X_hat = model_vae.predict(X)
    error = np.mean(np.abs(X - X_hat), axis=(1, 2, 3))
    errors[name] = error

##  Plot histograme using seaborn...
sns.set(style='whitegrid')
plt.figure(figsize=(12, 6))
plt.title('Reconstruction Error (MAE)')
plt.xlabel('Error')
plt.ylabel('Density')
for name in names:
    plt.hist(errors[name], bins=50, label=name, density=True, lw=0, alpha=0.5)
plt.legend()
plt.savefig('mh_reconstruction_error_hist.pdf')
plt.show()
plt.close()

##  Plot boxplot using seaborn...
df_dict = {'ds': [], 'error': []}
for name in names:
    df_dict['ds'].extend([name] * len(errors[name]))
    df_dict['error'].extend(errors[name])
df = pd.DataFrame(df_dict)
sns.set(style='whitegrid', font_scale=2)
plt.figure(figsize=(12, 6))
ax = sns.boxplot(data=df, x='ds', y='error', palette='tab10')
plt.ylabel('Reconstruction Error (MAE)')
plt.xlabel('')
plt.xticks(ticks=range(len(names)), labels=[r'$\mathbf{D}_{\mathbf{real, train}}$', r'$\mathbf{D}_{\mathbf{real, test}}$', r'$\mathbf{D}_{\mathbf{sim}}$', r'SAEVAE', r'cycleG', r'styleT'])
##  Change yticks to 4 decimal places...
locs, labels = plt.yticks()
new_labels = [f'{loc:.3f}' for loc in locs]
plt.yticks(locs, new_labels)
ax.set_xticklabels(ax.get_xticklabels(), weight='bold')
ax.set_yticklabels(ax.get_yticklabels(), weight='bold')
plt.savefig('mh_reconstruction_error_boxplot.pdf')
plt.show()
plt.close()

##  Statistical Tests...
errors = {name: errors[name][:min_length] for name in names}
df_stat_dict = {name: [] for name in names if name not in ['DS_real,train', 'DS_real,test']}
for name in names:
    if name in ['DS_real,train', 'DS_real,test']:
        continue
    print(f'Processing {name}...')
    p_value, a_12 = statisticalTests(errors['DS_real,train'], errors[name])
    print(f'p_value: {p_value}, a_12: {a_12}')
    df_stat_dict[name].extend([p_value, a_12])
    
df_stat = pd.DataFrame(df_stat_dict, index=['p_value', 'a_12'])
df_stat.to_csv('mh_reconstruction_stat.csv')
df_stat.to_latex('mh_reconstruction_stat.tex', float_format='%.3f')
