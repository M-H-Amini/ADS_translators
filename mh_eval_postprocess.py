import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging as log
from mh_ds import loadDataset
from git_dave2_model import loadModel as loadModelGit
from functools import partial
import os
import re

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def postprocess(model_name='dave2'):
    model_folders = []
    model_names = []
    ##  Loading models...
    for model in os.listdir('models'):
        os.path.isdir((model_folder := os.path.join('models', model))) and (model_folders.append(model_folder) or model_names.append(model_folder[10:]))

    model_folders.extend(['mh_cnn_udacity', 'mh_chauffeur_udacity', 'mh_epoch_udacity', 'mh_autumn_udacity', 'mh_dave2_beamng'])
    model_names.extend(['CNNUdacity', 'ChauffeurUdacity', 'EpochUdacity', 'AutumnUdacity', 'Dave2BeamNG'])

    df_eval = pd.read_csv(f'eval_dave2_offline.csv')
    df_eval.columns=['Model', 'UdacityJungle', 'BeamNG', 'SAEVAE']
    df_eval.set_index('Model', inplace=True)
    pattern = re.compile(f'{model_name}_udacity_[\d]+')
    df_filtered = df_eval[df_eval.index.str.contains(pattern)]

    df_filtered['Diff'] = df_filtered['BeamNG'] - df_filtered['SAEVAE']
    df_filtered.sort_values(by='Diff', inplace=True, ascending=False)
    df_filtered.to_csv(f'eval_{model_name}_offline_filtered.csv', float_format='%.3f')
    df_filtered.to_latex(f'eval_{model_name}_offline_filtered.tex', float_format='%.3f')

model = 'autumn'  ##  Can be ['dave2', 'cnn', 'chauffeur', 'epoch', 'autumn']
postprocess(model)