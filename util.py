
import os
import sys
import json
import random
from ast import literal_eval
from model import Transformer
from config import get_default_config

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
    altdata = data[:,:CL+1,:]
    _, hidden_states = model.forward_hs(altdata[:,:-1,:])
    hidden_states = torch.stack(hidden_states)
    hidden_states = hidden_states.transpose(0, 1)
    hidden_states = hidden_states[:, layer, neuron, :]

    return hidden_states
