import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging as log
import tensorflow_datasets as tfds
from tqdm import tqdm 
import cv2
import os

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
 
def loadDataset(dataset='udacity', resize=True):
    if dataset == 'udacity':
        from mh_dave2_data import prepareDataset as prepareDatasetUdacity
        x_transform = lambda x: x / 255.
        X_train, y_train, X_val, y_val, X_test, y_test = prepareDatasetUdacity('ds_udacity' , test_size=0.1, val_size=0.1, random_state=28, x_transform=x_transform, show=False)
        return X_train, y_train, X_val, y_val, X_test, y_test
    elif dataset == 'beamng':
        from mh_beamng_ds import prepareDataset as prepareDatasetBeamNG
        x_transform, y_transform = lambda x: x / 255., lambda y: y/0.11
        X_train, y_train, X_val, y_val, X_test, y_test = prepareDatasetBeamNG('ds_beamng', test_size=0.1, val_size=0.1, random_state=28, x_transform=x_transform, y_transform=y_transform, show=False)
        return X_train, y_train, X_val, y_val, X_test, y_test
    elif dataset == 'cycle':
        x_transform, y_transform = lambda x: x / 255., lambda y: y/0.11
        X_train, y_train, X_val, y_val, X_test, y_test = prepareDatasetBeamNG('ds_beamng_cycle', test_size=0.1, val_size=0.1, random_state=28, x_transform=x_transform, y_transform=y_transform, show=False)
        return X_train, y_train, X_val, y_val, X_test, y_test
    elif dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = np.concatenate((X_train[..., np.newaxis], X_train[..., np.newaxis], X_train[..., np.newaxis]), axis=-1)
        X_test = np.concatenate((X_test[..., np.newaxis], X_test[..., np.newaxis], X_test[..., np.newaxis]), axis=-1)
        X_train = np.array([tf.image.resize(x, (160, 320)).numpy() / 255. for x in X_train])
        X_test = np.array([tf.image.resize(x, (160, 320)).numpy() / 255. for x in X_test])
        return X_train, y_train, None, None, X_test, y_test
    elif dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        X_train = np.array([tf.image.resize(x, (66, 200)).numpy() / 255. for x in X_train])
        X_test = np.array([tf.image.resize(x, (66, 200)).numpy() / 255. for x in X_test])
        return X_train, y_train, None, None, X_test, y_test
    elif dataset == 'cats_vs_dogs':
        split = ['train[:70%]', 'train[70%:]']
        ds_train, ds_test = tfds.load(name='cats_vs_dogs', split=split, as_supervised=True)
        ds_train = ds_train.map(lambda x, y: (tf.image.resize(x, (66, 200)), y))
        ds_test = ds_test.map(lambda x, y: (tf.image.resize(x, (66, 200)), y))
        X_train = np.array([x.numpy()/255. for x, y in ds_train])
        y_train = np.array([y.numpy() for x, y in ds_train])
        X_test = np.array([x.numpy()/255. for x, y in ds_test])
        y_test = np.array([y.numpy() for x, y in ds_test])
        return X_train, y_train, None, None, X_test, y_test
    elif dataset == 'dclgan':
        folder = 'ds_dclgan/images/fake_A'
        csv_file = 'ds_dclgan/labels.csv'
        df = pd.read_csv(csv_file)
        df['img'] = df['img'].apply(lambda x: os.path.basename(x))
        transform_b = lambda x: x[130-66:130, 50:250, :] / 255.
        X_train, y_train = [], []
        for i in tqdm(os.listdir(folder)):
            if i.endswith('.png'):
                img = cv2.imread(os.path.join(folder, i))[..., ::-1]
                img = transform_b(img)
                X_train.append(img)
                steer = df[df['img'] == i[:-4] + '.jpg']['steer'].values[0]
                y_train.append(steer)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = X_train[-100:]
        y_test = y_train[-100:]
        X_train = X_train[:-100]
        y_train = y_train[:-100]
        # Normalizing the y arrays between -1 and +1...
        y_min = min(y_train.min(), y_test.min())
        y_max = max(y_train.max(), y_test.max())
        y_train = (y_train - y_min) / (y_max - y_min) * 2 - 1
        y_test = (y_test - y_min) / (y_max - y_min) * 2 - 1
        return X_train, y_train, None, None, X_test, y_test
    elif dataset == 'saevae':
        folder = 'ds_beamng_saevae'
        csv_file = 'ds_beamng_saevae/labels.csv'
        df = pd.read_csv(csv_file)
        df['img'] = df['img'].apply(lambda x: os.path.basename(x))
        transform_b = lambda x: x / 255.
        X_train, y_train = [], []
        for i in tqdm(os.listdir(folder)):
            if i.endswith('.jpg'):
                img = cv2.imread(os.path.join(folder, i))[..., ::-1]
                img = transform_b(img)
                X_train.append(img)
                steer = df[df['img'] == i[:-4] + '.jpg']['steer'].values[0]
                y_train.append(steer)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = X_train[-100:]
        y_test = y_train[-100:]
        X_train = X_train[:-100]
        y_train = y_train[:-100]
        # Normalizing the y arrays between -1 and +1...
        y_min = min(y_train.min(), y_test.min())
        y_max = max(y_train.max(), y_test.max())
        y_train = (y_train - y_min) / (y_max - y_min) * 2 - 1
        y_test = (y_test - y_min) / (y_max - y_min) * 2 - 1
        return X_train, y_train, None, None, X_test, y_test
    elif dataset == 'style':
        folder = 'ds_beamng_style'
        csv_file = 'ds_beamng_style/labels.csv'
        df = pd.read_csv(csv_file)
        df['img'] = df['img'].apply(lambda x: os.path.basename(x))
        transform_b = lambda x: x / 255.
        X_train, y_train = [], []
        for i in tqdm(os.listdir(folder)):
            if i.endswith('.png'):
                img = cv2.resize(cv2.imread(os.path.join(folder, i))[..., ::-1], (320, 160))
                img = transform_b(img)
                X_train.append(img)
                steer = df[df['img'] == i[:-4] + '.png']['steer'].values[0]
                y_train.append(steer)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = X_train[-100:]
        y_test = y_train[-100:]
        X_train = X_train[:-100]
        y_train = y_train[:-100]
        # Normalizing the y arrays between -1 and +1...
        y_min = min(y_train.min(), y_test.min())
        y_max = max(y_train.max(), y_test.max())
        y_train = (y_train - y_min) / (y_max - y_min) * 2 - 1
        y_test = (y_test - y_min) / (y_max - y_min) * 2 - 1
        return X_train, y_train, None, None, X_test, y_test

def plotHistogram(y, bins=100, title=None, output=None, show=True):
    plt.figure(figsize=(10, 5))
    plt.hist(y, bins=bins)
    title and plt.title(title)
    output and plt.savefig(output)
    show and plt.show()

def visualize(X1, X2, output=None, show=True):
    h, w = X1.shape[1:3]
    img = np.zeros((h*4, w*8, 3))
    for i in range(4):
        for j in range(4):
            img[i*h:(i+1)*h, 2*j*w:(2*j+1)*w, :] = X1[i*4+j]
            img[i*h:(i+1)*h, (2*j+1)*w:(2*j+2)*w, :] = X2[i*4+j]
    plt.figure(figsize=(20, 10))
    plt.imshow(img)
    show and plt.show()
    output and plt.imsave(output, img)
    
    
if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = loadDataset('udacity')
    print(f'Udaicty: X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')
    print(f'Udaicty: X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}')
    print(f'Udaicty: X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')
    X_train, y_train, X_val, y_val, X_test, y_test = loadDataset('beamng')
    print(f'BeamNG: X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')
    print(f'BeamNG: X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}')
    print(f'BeamNG: X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')
    X_train, y_train, X_val, y_val, X_test, y_test = loadDataset('udacitybeamng')
    print(f'UdaictyBeamNG: X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')
    print(f'UdaictyBeamNG: X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}')
    print(f'UdaictyBeamNG: X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')
    # X_train, y_train, X_val, y_val, X_test, y_test = loadDataset('cycle')
    # X = np.concatenate([X_train, X_val, X_test])
    y = np.concatenate([y_train, y_test])
    # print(X_train.min(), X_train.max())
    # print('X_train.shape:', X_train.shape, 'y_train.shape:', y_train.shape)

    # X_train, y_train, _, _, X_test, y_test = loadDataset('cats_vs_dogs')
    # X_train, y_train, _, _, X_test, y_test = loadDataset('mnist')
    # X_train, y_train, _, _, X_test, y_test = loadDataset('fake_gan')
    # X_train, y_train, _, _, X_test, y_test = loadDataset('saevae')
    
    plotHistogram(y, bins=100, output='hist_beamng_scaled.pdf', show=True)

    # print('y_train.shape:', y_train.shape)
    # print('X_test.shape:', X_test.shape)
    # print('y_test.shape:', y_test.shape)

