
import os
import sys
import json
import random
from ast import literal_eval
from model import Transformer
from config import get_default_config
from scipy.stats import linregress
import numpy as np
import torch

# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(config):
    """ monotonous bookkeeping """
    work_dir = config.system.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    # log the config itself
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))



def load_model(file='models/spring_16emb_2layer_10CL_20000epochs_0.001lr_64batch_model.pth', config = None):
    if config is None:
        config = get_default_config()
    model = Transformer(config)
    model_state = torch.load(file)
    model.load_state_dict(model_state)
    model.eval()
    return model


def get_hidden_state_old(model, CL, layer = 0, neuron = 0, target = 'omegas'):
    datadict = torch.load('data/spring_data.pth')
    data = torch.cat((datadict['sequences_test'], datadict['sequences_train']), dim=0)[:,:CL+1,:]
    omegas = torch.cat((datadict['test_omegas'], datadict['train_omegas']), dim=0)
    times = torch.cat((datadict['test_times'], datadict['train_times']), dim=0)[:, :CL+1]
    deltat = times[:,1] - times[:,0]
    target_dict = {'omegas':omegas, 'deltat':deltat}
    target_vals = target_dict[target]

    _, hidden_states = model.forward_hs(data[:,:-1,:])
    hidden_states = torch.stack(hidden_states)
    hidden_states = hidden_states.transpose(0, 1)
    hidden_states = hidden_states[:, layer, neuron, :]

    return hidden_states, target_vals

def get_hidden_state(model, data, CL, layer = 0, neuron = 0):
    hidden_states = get_hidden_states(model, data, CL)
    hidden_states = hidden_states[:, layer, neuron, :]

    return hidden_states

def get_hidden_states(model, data, CL):
    altdata = data[:,:CL+1,:]
    _, hidden_states = model.forward_hs(altdata[:,:-1,:])
    hidden_states = torch.stack(hidden_states)
    hidden_states = hidden_states.transpose(0, 1)
    return hidden_states

def euler_to_term(novar = False):
    mapping = {'x0x': 'x',
               'x1v': '\Delta t v',
               'x2v': '\Delta t^2 \gamma v',
               'x2x': '\Delta t^2 \omega_0^2 x',
               'x3v': '\Delta t^3 (4\gamma^2 - \omega_0^2) v',
               'x3x': '\Delta t^3 \gamma \omega_0^2 x',
               'x4v': '\Delta t^4 (-2 \gamma^3 + \omega_0^2 \gamma) v',
               'x4x': '\Delta t^4 (-4 \gamma^2 \omega_0^2 + \omega_0^4) x',
               'v0v': 'v',
               'v1v': '\Delta t \gamma v',
               'v1x': '\Delta t \omega_0^2 x',
               'v2v': '\Delta t^2 (4 \gamma^2 - \omega_0^2) v',
               'v2x': '\Delta t^2 \gamma \omega_0^2 x',
               'v3v': '\Delta t^3 (-2 \gamma^3 + \omega_0^2 \gamma) v',
               'v3x': '\Delta t^3 (-4 \gamma^2 \omega_0^2 + \omega_0^4) x',
               'v4v': '\Delta t^4 (16 \gamma^4 - 12 \omega_0^2 \gamma^2 + \omega_0^4) v',
               'v4x': '\Delta t^4 (2 \gamma^2 \omega_0^2 - \omega_0^4 \gamma) x'}
    mappingnovar =  {
               'x1v': '\Delta t',
               'x2v': '\Delta t^2 \gamma',
               'x2x': '\Delta t^2 \omega_0^2',
               'x3v': '\Delta t^3 (4\gamma^2 - \omega_0^2)',
               'x3x': '\Delta t^3 \gamma \omega_0^2',
               'x4v': '\Delta t^4 (-2 \gamma^3 + \omega_0^2 \gamma)',
               'x4x': '\Delta t^4 (-4 \gamma^2 \omega_0^2 + \omega_0^4)',

               'v1v': '\Delta t \gamma',
               'v1x': '\Delta t \omega_0^2',
               'v2v': '\Delta t^2 (4 \gamma^2 - \omega_0^2)',
               'v2x': '\Delta t^2 \gamma \omega_0^2',
               'v3v': '\Delta t^3 (-2 \gamma^3 + \omega_0^2 \gamma)',
               'v3x': '\Delta t^3 (-4 \gamma^2 \omega_0^2 + \omega_0^4)',
               'v4v': '\Delta t^4 (16 \gamma^4 - 12 \omega_0^2 \gamma^2 + \omega_0^4)',
               'v4x': '\Delta t^4 (2 \gamma^2 \omega_0^2 - \omega_0^4 \gamma)'}
    if novar:
        return mappingnovar
    else:
        return mapping
    
def matrix_weights_to_term():
    mapping = {'w00': '(\cos {\omega \Delta t} + \gamma / \omega \sin {\omega \Delta t})x',
               'w01': ' (\sin { \omega \Delta t })v ',
               'w10': '-({\gamma^2+\omega^2}) / {\omega} \sin {\omega \Delta t}x',
               'w11': '(\cos {\omega \Delta t} - \gamma / \omega \sin {\omega \Delta t})v'}
    return mapping



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


def linear_multistep_coefficients(explicit):
    if explicit:
        all_coefficients = {
                1: [[1], [1]],
                2: [[1], [3/2, -1/2]],
                3: [[1], [23/12, -16/12, 5/12]],
                4: [[1], [55/24, -59/24, 37/24, -9/24]],
                5: [[1], [1901/720, -2774/720, 2616/720, -1274/720, 251/720]],
                6: [[1], [4277/1440, -7923/1440, 9982/1440, -7298/1440, 2877/1440, -475/1440]],
                7: [[1], [198721/60480, -447288/60480, 705549/60480, -688256/60480, 407139/60480, -134472/60480, 19087/60480]],
                8: [[1], [434241/120960, -1152169/120960, 2183877/120960, -2664477/120960, 2102243/120960, -1041723/120960, 295767/120960, -36799/120960]],
                9: [[1], [14097247/3628800, -43125206/3628800, 95476786/3628800, -139855262/3628800, 137968480/3628800, -91172642/3628800, 38833486/3628800, -9664106/3628800, 1070017/3628800]],
                10: [[1], [30277247/7257600, -104995189/7257600, 265932123/7257600, -454661776/7257600, 538363838/7257600, -444772162/7257600, 252618224/7257600, -94307320/7257600, 20884811/7257600, -2082753/7257600]]
        }
    else:
        all_coefficients = {
                1: [[0, 1], [1]],
                2: [[0, 1], [1/2, 1/2]],
                3: [[0, 1], [5/12, 2/3, -1/12]],
                4: [[0, 1], [9/24, 19/24, -5/24, 1/24]],
                5: [[0, 1], [251/720, 646/720, -264/720, 106/720, -19/720]],
                6: [[0, 1], [475/1440, 1427/1440, -798/1440, 482/1440, -173/1440, 27/1440]],
                7: [[0, 1], [19087/60480, 65112/60480, -46461/60480, 37504/60480, -20211/60480, 6312/60480, -863/60480]],
                8: [[0, 1], [36799/120960, 139849/120960, -121797/120960, 123133/120960, -88547/120960, 41499/120960, -11351/120960, 1375/120960]],
                9: [[0, 1], [1070017/3628800, 4467094/3628800, -4604594/3628800, 5595358/3628800, -5033120/3628800, 3146338/3628800, -1291214/3628800, 312874/3628800, -33953/3628800]],
                10: [[0, 1], [2082753/7257600, 9449717/7257600, -11271304/7257600, 16002320/7257600, -17283646/7257600, 13510082/7257600, -7394032/7257600, 2687864/7257600, -583435/7257600, 57281/7257600]]
            }
    return all_coefficients