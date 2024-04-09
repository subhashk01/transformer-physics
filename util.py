from config import get_default_config
from model import Transformer
import torch
import numpy as np
from scipy.stats import linregress

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
