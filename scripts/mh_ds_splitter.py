##  Description: Train and Test splitter for training CycleGAN, DCLGAN models.

from mh_dave2_data import prepareDataset as prepareDatasetUdacity
from mh_beamng_ds import prepareDataset as prepareDatasetBeamNG
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

def saveDataset(X_train, y_train, X_test, y_test, train_folder, test_folder):
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    ds_dict = {'img': [], 'steer': []}
    for i in tqdm(range(len(X_train))):
        plt.imsave(os.path.join(train_folder, 'img_train_{:05d}.jpg'.format(i)), X_train[i])
        ds_dict['img'].append(os.path.join(train_folder, 'img_train_{:05d}.jpg'.format(i)))
        ds_dict['steer'].append(y_train[i])
    for i in tqdm(range(len(X_test))):
        plt.imsave(os.path.join(test_folder, 'img_test_{:05d}.jpg'.format(i)), X_test[i])
        ds_dict['img'].append(os.path.join(test_folder, 'img_test_{:05d}.jpg'.format(i)))
        ds_dict['steer'].append(y_test[i])
    ds_df = pd.DataFrame(ds_dict)
    return ds_df


os.makedirs('ds_sim', exist_ok=True)


split_ratio = 0.8

##  Udacity Jungle...
dataset_folder = 'UdacityDS/self_driving_car_dataset_jungle/IMG'
dataset_csv = 'UdacityDS/self_driving_car_dataset_jungle/driving_log.csv'
train_cols = ['center']
# train_cols = ['center', 'left', 'right']
transform_u = lambda x: x[70:136, 100:300, :] / 255.
X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = prepareDatasetUdacity(dataset_folder, dataset_csv, train_cols, reduce_ratio=0.7, test_size=0.1, val_size=0.1, transform=transform_u, show=False)
X_test = np.concatenate((X_val, X_test), axis=0)
y_test = np.concatenate((y_val, y_test), axis=0)
df_A = saveDataset(X_train, y_train, X_test, y_test, 'ds_sim/trainA', 'ds_sim/testA')

##  BeamNG...
transform_b = lambda x: x[130-66:130, 60:260, :]  / 255.
json_folder = 'ds_beamng'
test_size = 0.1
val_size = 0.1
step = 15
X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = prepareDatasetBeamNG(json_folder, step=step, test_size=test_size, val_size=val_size, random_state=28, transform=transform_b, show=False)
X_test = np.concatenate((X_val, X_test), axis=0)
y_test = np.concatenate((y_val, y_test), axis=0)
df_B = saveDataset(X_train, y_train, X_test, y_test, 'ds_sim/trainB', 'ds_sim/testB')

##  Concatenate...
df = pd.concat([df_A, df_B], axis=0)
df.to_csv('ds_sim/ds_sim.csv', index=False)


