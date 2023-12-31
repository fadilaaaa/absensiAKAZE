import pandas as pd
import pickle
import os
import cv2
import numpy as np


dir_path = os.path.dirname(os.path.realpath(__file__))
pathDB = f'{dir_path}/../data/faceDB.csv'


def get_data():
    df = pd.read_csv(pathDB)
    return df.to_list()


def get_attendances():
    df = pd.read_csv(pathDB)
    return df['name'].values.tolist()


def save_data(name, desc=None, img=None):
    df = pd.read_csv(pathDB)
    filename = None
    if desc is not None:
        filename = f'{name}.pkl'
        with open(f'{dir_path}/../data/{filename}', 'wb') as f:
            pickle.dump(desc, f)
        if img is not None:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{dir_path}/../data/image/{name}.png', img)
            print("name", name)
    if name not in df['name'].values:
        df.loc[len(df.index)] = [name, filename]
    if name in df['name'].values:
        df.loc[df['name'] == name, 'desc'] = filename
    df.to_csv(pathDB, index=False)
