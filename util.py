from config import get_default_config
from model import Transformer
import torch
import numpy as np
from scipy.stats import linregress
import pandas as pd
import random   


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(df_row):
    # expects row from model dataframe
    config = get_default_config()
    config.n_embd = df_row['emb']
    config.n_layer = df_row['layer']
    config.max_seq_length = df_row['CL']
    file = df_row['modelpath']

    model = Transformer(config)
    model_state = torch.load(file)
    model.load_state_dict(model_state)
    model.eval()
    return model

def get_data(datatype = 'underdamped', traintest = 'train'):
    data = torch.load('data/dampedspring_data.pth')
    gammas = data[f'gammas_{traintest}_{datatype}']
    omegas = data[f'omegas_{traintest}_{datatype}']
    sequences = data[f'sequences_{traintest}_{datatype}']
    times = data[f'times_{traintest}_{datatype}']
    deltat = times[:, 1] - times[:, 0]
    return gammas, omegas, sequences, times, deltat

def get_log_log_linear(x,y):
    # gets the lienar relationship between logx and logy
    xlog = np.log(x)
    ylog = np.log(y)
    slope, intercept, r_value, p_value, std_err = linregress(xlog, ylog)
    return slope, intercept, r_value

def get_model_df(df, layer, emb):
    df_model = df[(df['m-layer'] == layer) & (df['m-emb'] == emb)]
    return df_model
def hello():
    print('hello')

def get_targetmethod_df(df, targetmethod):
    df_targetmethod = df[df['p-targetmethod'] == targetmethod]
    return df_targetmethod

def get_target_df(df, target):
    df_target = df[df['p-targetname'] == target]
    return df_target

def get_layer_pos(df, layerpos, inlayerpos):
    df_model = df[(df['h-layerpos'] == layerpos) & (df['h-inlayerpos'] == inlayerpos)]
    return df_model


if __name__ == '__main__':
    df = 'x'
    get_layer_pos(df, 0, 0)