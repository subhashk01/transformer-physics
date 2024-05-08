from config import get_default_config, linreg_config
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


def load_model(df_row, datatype):
    # expects row from model dataframe
    if 'linreg' in datatype:
        config = linreg_config()
        config.max_seq_length = df_row['CL']+1
    else:
        config = get_default_config()
        config.max_seq_length = df_row['CL']+1
    config.n_embd = df_row['emb']
    config.n_layer = df_row['layer']
    
    file = df_row['modelpath']
    return load_model_file(file, config)

def load_model_file(file, config):
    model = Transformer(config)
    model_state = torch.load(file, map_location=torch.device('cpu'))
    model.load_state_dict(model_state)
    model.eval()
    return model
  
    


def get_data(datatype = 'underdamped', traintest = 'train'):
    if 'linreg' in datatype:
        datadict = torch.load('data/linreg1_data.pth')
        data = datadict[f'{traintest}data'].unsqueeze(-1)
        weights = datadict[f'w_{traintest}']
        return weights, data
    elif 'undamped' in datatype:
        data = torch.load('data/undampedspring_data.pth')
        sequences = data[f'sequences_{traintest}']
        omegas = data[f'{traintest}_omegas']
        times = data[f'{traintest}_times']
        deltat = times[:, 1] - times[:, 0]
        gammas = torch.zeros(omegas.shape)
        return gammas, omegas, sequences, times, deltat
    elif 'damped' in datatype:
        data = torch.load('data/dampedspring_data.pth')
        gammas = data[f'gammas_{traintest}_{datatype}']
        omegas = data[f'omegas_{traintest}_{datatype}']
        sequences = data[f'sequences_{traintest}_{datatype}']
        times = data[f'times_{traintest}_{datatype}']
        deltat = times[:, 1] - times[:, 0]
        return gammas, omegas, sequences, times, deltat

def get_log_log_linear(x,y, return_mse = False):
    # gets the lienar relationship between logx and logy
    xlog = np.log(x)
    ylog = np.log(y)
    slope, intercept, r_value, p_value, std_err = linregress(xlog, ylog)
    if return_mse:
        # Predict ylog using the linear model
        ylog_pred = slope * xlog + intercept
        # Calculate MSE
        mse = np.mean((ylog - ylog_pred) ** 2)
        return slope, intercept, r_value, mse
    return slope, intercept, r_value

def get_mtype_dtype_ttype(df, modeltype, datatype, traintest):
    df_mtype = df[df['m-modeltype'] == modeltype]
    df_dtype = df_mtype[df_mtype['h-datatype'] == datatype]
    df_ttype = df_dtype[df_dtype['h-traintest'] == traintest]
    return df_ttype

def get_model_df(df, layer, emb):
    df_model = df[(df['m-layer'] == layer) & (df['m-emb'] == emb)]
    return df_model



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
    #get_layer_pos(df, 0, 0)