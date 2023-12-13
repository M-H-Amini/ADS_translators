import logging as log
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import json
import os
import shutil
import cv2
from mh_dave2_data import plotHistogram, splitDataset, imageGenerator, visualizeGenerator
from sklearn.model_selection import train_test_split

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def json2CSV(json_folder, step=5):
    """
    Reads the json files in the given folder and creates a pandas DataFrame containing the dataset.
    The resulting DataFrame has the following columns:
    - img: the image file name
    - steering_input: the steering angle input value
    - throttle_input: the throttle input value
    - brake_input: the brake input value
    - is_oob: a boolean indicating whether the vehicle is out of bounds
    The function drops the 'is_oob' and 'img' columns from the DataFrame, converts the other columns to float32,
    and drops the rows where the absolute value of the steering angle input is greater than 0.11.
    Finally, the function saves the resulting DataFrame to a CSV file in the same folder as the input json files.
    
    Args:
    - json_folder (str): the path to the folder containing the input json files
    - step (int): the step to use when reading the json files (default: 5)
    
    Returns:
    - df (pandas.DataFrame): the resulting DataFrame containing the dataset
    """
    jsons = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    log.debug(f'Found {len(jsons)} json files in {json_folder}')
    cols = list(json.load(open(os.path.join(json_folder, jsons[0]))).keys())
    df = pd.DataFrame(columns=cols)
    for json_file in tqdm(jsons):
        json_path = os.path.join(json_folder, json_file)
        df_json = pd.read_json(json_path)
        ##  To remove first and last 10% of the data (black frames and grass frames)
        a, b = int(0.1 * len(df_json)), int(0.9 * len(df_json))
        df_json = df_json.iloc[a:b:step]
        ##  Concatenating df_json to df...
        df = pd.concat([df, df_json], ignore_index=True)
    ##  Converting all columns other than is_oob and img to float32...
    for col in df.columns:
        if col not in ['is_oob', 'img']:
            df[col] = df[col].astype(np.float32)
    ##  Dropping steering angles with absolute value more than 0.11 to have similar histogram to UdacityJungle...
    df = df[(df['steering_input'] > -0.11) & (df['steering_input'] < 0.11)]
    ##  Saving df to csv...
    output_csv = os.path.join(json_folder, 'ds_beamng.csv')
    df.to_csv(output_csv, index=False)
    log.info(f'The resulting csv file saved to {output_csv}!')
    return df

def readDataset(json_folder, step=15, transform=lambda x:x, output_folder=None, reduction_ratio=0.8):
    """Reads the dataset from the csv file. It also balances the dataset by flipping the images and steering angles.

    Args:
        json_folder (str): Path to json folder.
        step (int, optional): Step to skip frames. Defaults to 15.
        transform (function, optional): Transform function to be applied to images. Defaults to lambda x:x.

    Returns:
        df (pandas.DataFrame): Dataframe containing the dataset.
    """
    df = pd.read_csv(os.path.join(json_folder, 'ds_beamng.csv'))
    df['steering'] = df['steering_input']  ##  Renaming steering_input to steering to be consistent with Udacity dataset...
    df = df[['img', 'steering']]  ##  Dropping other columns...
    log.info(f'Reading {len(df)} images...')
    df[['img_path']] = df[['img']]
    df_flip = df.copy()  ##  Coppying the dataframe to flip it later...
    df['img'] = df['img'].apply(lambda x: np.asarray(Image.open(os.path.join(json_folder, 'img', x))))
    df_flip['img'] = df_flip['img'].apply(lambda x: np.fliplr(np.asarray(Image.open(os.path.join(json_folder, 'img', x)))))
    df_flip['steering'] = -df_flip['steering']
    if reduction_ratio:
        df_non_zero = df[(df['steering'] > 0.0253) | (df['steering'] < -0.0253)]
        df_zero = df[(df['steering'] <= 0.0253) | (df['steering'] >= -0.0253)]
        df_zero = df_zero.sample(frac=1-reduction_ratio).reset_index(drop=True)
        df = pd.concat([df_non_zero, df_zero], ignore_index=True)
        df_flip_non_zero = df_flip[(df_flip['steering'] > 0.0253) | (df_flip['steering'] < -0.0253)]
        df_flip_zero = df_flip[(df_flip['steering'] <= 0.0253) | (df_flip['steering'] >= -0.0253)]
        df_flip_zero = df_flip_zero.sample(frac=1-reduction_ratio).reset_index(drop=True)
        df_flip = pd.concat([df_flip_non_zero, df_flip_zero], ignore_index=True)
    if output_folder:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
        for i in tqdm(range(len(df))):
            img = Image.fromarray(df['img'].iloc[i])
            img.save(os.path.join(output_folder, 'images', os.path.basename(df['img_path'].iloc[i])))
            df['img_path'].iloc[i] = os.path.basename(df['img_path'].iloc[i])
        for i in tqdm(range(len(df_flip))):
            img = Image.fromarray(df_flip['img'].iloc[i])
            img.save(os.path.join(output_folder, 'images', os.path.basename(df_flip['img_path'].iloc[i])[:-4]+'_flip.png'))
            df_flip['img_path'].iloc[i] = os.path.basename(df_flip['img_path'].iloc[i])[:-4]+'_flip.png'
        df = pd.concat([df, df_flip], ignore_index=True)
        df = df.drop(columns=['img'])
        df.to_csv(os.path.join(output_folder, 'ds_beamng.csv'), index=False)
    return df


def prepareDataset(dataset_folder, test_size=0.2, val_size=0.2, random_state=28, x_transform=lambda x:x, y_transform=lambda y:y, show=True):
    df = pd.read_csv(os.path.join(dataset_folder, 'ds_beamng.csv'))
    log.info(f'Reading {len(df)} images...')
    X = np.array(list(map(lambda x: x_transform(cv2.imread(os.path.join(dataset_folder, 'images', x))), df['img_path'].values)))[...,::-1]
    y = df['steering'].values
    y = np.array(list(map(lambda y: y_transform(y), y)))
    df['steering'] = y
    log.info(f'X shape: {X.shape}, y shape: {y.shape}')
    show and plotHistogram(df, 'steering', output_name='ds_beamng_hist.pdf', show=show)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size / (1 - test_size), random_state=random_state)
    log.info(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
    log.info(f'X_val shape: {X_val.shape}, y_val shape: {y_val.shape}')
    log.info(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
    df.to_csv(os.path.join(dataset_folder, 'ds_beamng_normalized.csv'), index=False)
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == '__main__':
    ##  Uncomment the following lines to generate the dataset (with flipped images) in ds_beamng folder...
    json_folder = 'ds_beamng_raw'
    json2CSV(json_folder, step=15)
    df = readDataset(json_folder, step=15, output_folder='ds_beamng', reduction_ratio=0.9)

    ##  Reading the dataset from ds_beamng folder...
    y_transform = lambda y: y/0.11
    X_train, y_train, X_val, y_val, X_test, y_test = prepareDataset('ds_beamng', test_size=0.1, val_size=0.1, random_state=28, y_transform=y_transform, show=True)
    gen_train = imageGenerator(X_train, y_train, (0.6, 1.), batch_size=16, shuffle=True)
    visualizeGenerator(gen_train)
    