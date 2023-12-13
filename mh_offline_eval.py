import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging as log
from mh_ds import loadDataset
from git_dave2_model import loadModel as loadModelGit
from functools import partial
import os
from mh_stat_test import statisticalTests

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##  GPU...
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    log.info(f'GPUs: {gpus}')
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
    except RuntimeError as e:
        log.error(e)
else:
    log.info('No GPUs found')

def loadModel(model='git'):
    if 'git' in model.lower():
        return loadModelGit('DAVE2-Keras-master/model.h5')
    else:
        return load_model(model)

def evaluate(train, val, test, model, verbose=True):
    res = {}
    if train:
        loss_train = model.evaluate(train[0], train[1], steps=len(train[0])//128)
        pred_train = model.predict(train[0], steps=len(train[0])//128)[:, 0]
        verbose and log.info(f'Train Loss: {loss_train}')
        res['train'] = loss_train
        res['train_error'] = np.abs(pred_train - train[1])
    if val:
        loss_val = model.evaluate(val[0], val[1], steps=len(val[0])//128)
        pred_val = model.predict(val[0], steps=len(val[0])//128)[:, 0]
        verbose and log.info(f'Val Loss: {loss_val}')
        res['val'] = loss_val
        res['val_error'] = np.abs(pred_val - val[1])
    if test:
        loss_test = model.evaluate(test[0], test[1], steps=len(test[0])//128)
        pred_test = model.predict(test[0], steps=len(test[0])//128)[:, 0]
        verbose and log.info(f'Test Loss: {loss_test}')
        res['test'] = loss_test
        res['test_error'] = np.abs(pred_test - test[1])
    return res
 

##  Datasets...
ds_names = ['udacity', 'beamng', 'saevae', 'cycle', 'style']  #  , 'dclgan', 'saevae', 'magenta']
df_index = ['DS_real', 'DS_sim', 'SAEVAE', 'cycleG', 'styleT']
# ds_names = ['udacity', 'beamng']  #  , 'dclgan', 'saevae', 'magenta']
# df_index = ['DS_real', 'DS_sim']
ds = [loadDataset(ds_name) for ds_name in ds_names]
func_map = lambda x: {'total': (np.concatenate((x[0], x[4]), axis=0), np.concatenate((x[1], x[5]), axis=0)), 'test': (x[4], x[5])}
ds = map(func_map, ds)
ds = {ds_name: ds_ for ds_name, ds_ in zip(ds_names, ds)}
log.info('Evaluating models...')

model_folders = []
model_names = []

model_folders.extend(['mh_chauffeur_udacity', 'mh_epoch_udacity', 'models/mh_autumn_udacity_1', 'mh_dave2_udacity'])
model_names.extend(['ChauffeurUdacity', 'EpochUdacity', 'AutumnUdacity', 'Dave2Udacity'])
# model_folders.extend(['mh_chauffeur_udacity'])
# model_names.extend(['ChauffeurUdacity'])

models = {key: loadModel(key) for key in model_folders}
df_eval = {key: [] for key in models.keys()}
errors = {ds : [] for ds in ds_names}

for i, model_key in enumerate(models.keys()):
    model = models[model_key]
    model_name = model_names[i]
    for j, ds_name in enumerate(ds.keys()):
        if 'beamng' in model_key.lower() and 'beamng' in ds_name.lower():  ##  To use test part of beamng dataset for beamng models
            print(f'{model_name} on test part of {ds_name}, shape: {ds[ds_name]["test"][0].shape}')
            res = evaluate(train=None, val=None, test=ds[ds_name]['test'], model=model, verbose=False)
            df_eval[model_key].append(res['test'])
            errors[ds_name].append(res['test_error'])
        else:
            print(f'{model_name} on total part of {ds_name}')
            res = evaluate(train=ds[ds_name]['total'], val=None, test=None, model=model, verbose=False)
            df_eval[model_key].append(res['train'])
            errors[ds_name].append(res['train_error'])

ds_name_map = {'udacity': 'DS_real', 'beamng': 'DS_sim', 'saevae': 'SAEVAE', 'cycle': 'cycleG', 'style': 'styleT'}
df_dict = {'dataset': [], 'error': []}
for ds_name in ds_names:
    errors[ds_name] = np.concatenate(errors[ds_name], axis=0)
    df_dict['dataset'].extend([ds_name_map[ds_name]]*len(errors[ds_name]))
    df_dict['error'].extend(errors[ds_name])

means = {ds_name: np.mean(errors[ds_name]) for ds_name in ds_names}


##  Draw boxplot for errors using seaborn...
df = pd.DataFrame(df_dict)
sns.set_theme(style="whitegrid", font_scale=1.5)
plt.figure(figsize=(10, 5))
ax = sns.boxplot(x="dataset", y="error", data=df, palette='tab10')
##  Display mean values on the plot...
for i, ds_name in enumerate(ds_names):
    plt.text(i, means[ds_name], f'{means[ds_name]:.3f}', horizontalalignment='center', size='medium', color='black', weight='semibold')
ax.set_xlabel('')
ax.set_ylabel('MAE')
plt.xticks(ticks=range(len(ds_names)), labels=[r'$\mathbf{D}_{\mathbf{real}}$', r'$\mathbf{D}_{\mathbf{sim}}$', r'SAEVAE', r'cycleG', r'styleT'])
ax.set_xticklabels(ax.get_xticklabels(), weight='bold')
ax.set_yticklabels(ax.get_yticklabels(), weight='bold')
plt.savefig('mh_offline_errors.pdf')
plt.close()

##  Statistical tests...
a = 'saevae'
bs = ['beamng', 'cycle', 'style']
df_stats = {'dataset': [], 'p_value': [], 'a12_value': []}
for b in bs:
    length = min(len(errors[b]), len(errors[a]))
    p_value, a12_value = statisticalTests(errors[b][:length], errors[a][:length])
    df_stats['dataset'].append(b)
    df_stats['p_value'].append(p_value)
    df_stats['a12_value'].append(a12_value)
    print(f'Statistical test between SAEVAE and {b}: p_value: {p_value}, a12_value: {a12_value}')
df_stats = pd.DataFrame(df_stats)
df_stats.to_latex('mh_offline_stat_tests.tex', float_format='%.3f')
df_stats.to_csv('mh_offline_stat_tests.csv', float_format='%.3f')


df_eval = pd.DataFrame(df_eval, index=df_index).T
df_eval.rename(index={model_folders[i]:model_names[i] for i in range(len(model_folders))}, inplace=True)

df_eval.to_latex('eval_offline.tex', float_format='%.3f')
df_eval.to_csv('eval_offline.csv', float_format='%.3f')
