from util import get_data
import torch
from math import factorial
import os
import pandas as pd
from sklearn.linear_model import Ridge
import time
import sys
from sklearn.cross_decomposition import CCA
from sklearn.metrics import mean_squared_error
import numpy as np
from analyze_models import get_model_hs_df, get_df_models
from tqdm import tqdm

def get_expanded_data(datatype, traintest):
    gammas, omegas, sequences, times, deltat = get_data(datatype, traintest)
    # tiles omegas and gammas to match the shape of sequences
    X, y = sequences[:,:-1], sequences[:,1:]
    x, v = X[:,:,0], X[:,:,1]
    omegas = omegas.repeat(X.shape[1], 1).T
    gammas = gammas.repeat(X.shape[1], 1).T
    deltat = deltat.repeat(X.shape[1], 1).T
    return omegas, gammas, deltat, X, y, x, v 

def generate_rk_targets(datatype, traintest, maxdeg = 5, reverse = False):
    prec = ''
    if reverse:
        prec = 'r'
    criterion = torch.nn.MSELoss()
    omegas, gammas, deltat, X, y, x, v = get_expanded_data(datatype, traintest)
    
    A = torch.zeros((X.shape[0], X.shape[1],2,2))
    A[:, :, 0, 1] = 1
    A[:, :, 1, 0] = -omegas**2
    A[:, :, 1, 1] = -2*gammas
    ccatargets = {}
    targets = {}
    # 5000 x 65 x 2 x 2 identity matrix
    totalmat = torch.eye(2).repeat(X.shape[0], X.shape[1], 1, 1)

    ccatargets[f'{prec}Adt00'] = torch.zeros((A.shape[0], A.shape[1], maxdeg))
    ccatargets[f'{prec}Adt01'] = torch.zeros_like(ccatargets[f'{prec}Adt00'])
    ccatargets[f'{prec}Adt10'] = torch.zeros_like(ccatargets[f'{prec}Adt00'])
    ccatargets[f'{prec}Adt11'] = torch.zeros_like(ccatargets[f'{prec}Adt00'])

    for deg in range(1,maxdeg+1):
        Ai = torch.matrix_power(A, deg)
        ccatargets[f'{prec}Adt00'][:,:,deg-1] = Ai[:,:,0,0] * deltat**deg
        ccatargets[f'{prec}Adt01'][:,:,deg-1] = Ai[:,:,0,1] * deltat**deg
        ccatargets[f'{prec}Adt10'][:,:,deg-1] = Ai[:,:,1,0] * deltat**deg
        ccatargets[f'{prec}Adt11'][:,:,deg-1] = Ai[:,:,1,1] * deltat**deg
        deltatdeg = deltat**deg
        deltatdeg = deltatdeg.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, 2)
        Adtdeg = Ai * deltatdeg
        targets[f'{prec}Adt{deg}'] = Adtdeg.view(Adtdeg.shape[0], Adtdeg.shape[1], Adtdeg.shape[2]*Adtdeg.shape[3])
        print(targets[f'{prec}Adt{deg}'].shape)

        coef = 1/factorial(deg)
    
        currentmat = coef * Ai * (deltat**deg).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, 2)
        totalmat += currentmat 
        totalmat = totalmat.squeeze(-1)
        ypred = (totalmat @ X.unsqueeze(-1)).squeeze(-1)
        loss = criterion(ypred, y)
        print(f"Loss for degree {deg}: {loss:.2e}")


    if reverse:
        for deg in range(2,maxdeg+1):
            tocat = [targets[f'{prec}Adt{i}'] for i in range(1, deg+1)]
            targets[f'{prec}Adt1{deg}'] = torch.concatenate(tocat, dim = -1)

    for key in targets.keys():
        print(targets[key].shape)
    
    ccafname = f'{prec}rk_cca_targets_deg{maxdeg}.pth'
    save_probetargets(ccatargets, ccafname, datatype, traintest)
    fname = f'{prec}rk_targets_deg{maxdeg}.pth'
    save_probetargets(targets, fname, datatype, traintest)
    return targets


def generate_exp_targets(datatype, traintest, reverse = False, maxdeg = 5):
    criterion = torch.nn.MSELoss()
    omegas, gammas, deltat, X, y, x, v = get_expanded_data(datatype, traintest)
    A = torch.zeros((X.shape[0],2,2)) # each time series has its own A. but A is constant with CL
    A[:, 0, 0] = 0
    A[:, 0, 1] = 1
    A[:, 1, 0] = -omegas[:, 0]**2
    A[:, 1, 1] = -2*gammas[:, 0]
    deltat = deltat[:,0]
    deltat = deltat.unsqueeze(-1).unsqueeze(-1).expand(-1, 2, 2)
    Adt = A * deltat
    eAdt = torch.matrix_exp(Adt)
    eAdt = eAdt.unsqueeze(1).expand(-1, X.shape[1], -1, -1)
    def test_exp():
        ypred = eAdt@X.unsqueeze(-1)
        ypred = ypred.squeeze(-1)
        mse = criterion(ypred, y)
        return mse
    targets = {}
    ccatargets = {}
    prec = ''
    if reverse:
        prec = 'r'

    ccatargets[f'{prec}eAdt00'] = torch.zeros(eAdt.shape[0], eAdt.shape[1], maxdeg)#eAdt[:, :, 0, 0]
    ccatargets[f'{prec}eAdt01'] = torch.zeros_like(ccatargets[f'{prec}eAdt00'])
    ccatargets[f'{prec}eAdt10'] = torch.zeros_like(ccatargets[f'{prec}eAdt00'])
    ccatargets[f'{prec}eAdt11'] = torch.zeros_like(ccatargets[f'{prec}eAdt00'])

    for deg in range(1,maxdeg+1): # raise eAdt to some power
        eAdtpow = torch.matrix_power(eAdt, deg)
        ccatargets[f'{prec}eAdt00'][:,:,deg-1] = eAdtpow[:,:,0,0]
        ccatargets[f'{prec}eAdt01'][:,:,deg-1] = eAdtpow[:,:,0,1]
        ccatargets[f'{prec}eAdt10'][:,:,deg-1] = eAdtpow[:,:,1,0]
        ccatargets[f'{prec}eAdt11'][:,:,deg-1] = eAdtpow[:,:,1,1]
        targets[f'{prec}eAdt{deg}'] = eAdtpow.view(eAdtpow.shape[0], eAdtpow.shape[1], eAdtpow.shape[2]*eAdtpow.shape[3])
    
    if reverse:
        for deg in range(2,maxdeg+1):
            tocat = [targets[f'{prec}eAdt{i}'] for i in range(1, deg+1)]
            targets[f'{prec}eAdt1{deg}'] = torch.concatenate(tocat, dim = -1)

    mse = test_exp()
    print(mse)
    ccafname = f'{prec}eA_cca_targets_deg{maxdeg}.pth'
    save_probetargets(ccatargets, ccafname, datatype, traintest)

    for key in targets.keys():
        print(targets[key].shape)

    fname = f'{prec}eA_targets_deg{maxdeg}.pth'
    save_probetargets(targets, fname, datatype, traintest)
    return targets

def generate_rkexp_targets_REVERSE(datatype, traintest, maxdeg = 5):
    exptargets = generate_exp_targets(datatype, traintest, reverse = True, maxdeg = maxdeg)
    rktargets = generate_rk_targets(datatype, traintest, reverse = True, maxdeg = maxdeg)

    targets = {}
    for deg in range(1, maxdeg+1):
        if deg == 1:
            key = '1'
        else: key = f'1{deg}'
        targets[f'rkeAdt1{deg}'] = torch.concatenate((exptargets[f'reAdt{key}'], rktargets[f'rAdt{key}']), dim = -1)
    for key in targets.keys():
        print(key, targets[key].shape)

    fname = f'rrkeA_targets_deg{maxdeg}.pth'
    save_probetargets(targets, fname, datatype, traintest)

def generate_lr_targets(datatype = 'linreg1', traintest = 'train'):
    # makes targets for linreg model, which are w, wx
    # ONLY WORKS FOR 1D W
    criterion = torch.nn.MSELoss()
    w, sequences = get_data(datatype, traintest)
    X, y = sequences[:,:-1], sequences[:,1:]
    X, y = X.squeeze(-1), y.squeeze(-1)
    w = w.repeat(X.shape[1], 1).T
    targets = {}
    for i in range(1, 6):
        targets[f'lr_w{i}'] = w**i
        targets[f'lr_w{i}x'] = targets[f'lr_w{i}'] * X
        targets[f'lr_w{i}x{i}'] = targets[f'lr_w{i}'] * X**i
    
    x = X[:,0::2]
    y = y[:,0::2]
    w = w[:,0::2]
    # make w same size as X
    ypred = w*x
    loss = criterion(ypred, y)
    print(f'prediction loss: {loss}')
    save_probetargets(targets, 'lr_targets.pth', datatype, traintest)
    return targets

def generate_lr_cca_targets(datatype = 'linreg1cca', traintest = 'train', maxdeg = 5, save = True):
    w, sequences = get_data(datatype, traintest)
    X, y = sequences[:,:-1], sequences[:,1:]
    X, y = X.squeeze(-1), y.squeeze(-1)
    w = w.repeat(X.shape[1], 1).T
    wpow = torch.zeros((w.shape[0], w.shape[1], maxdeg))
    wpowx = torch.zeros(wpow.shape)
    wxpow = torch.zeros(wpow.shape)
    for deg in range(1, maxdeg+1):
        wpow[:, :, deg - 1] = w**deg
        wpowx[:, :, deg - 1] = wpow[:, :, deg - 1] * X
        wxpow[:, :, deg - 1] = wpow[:, :, deg - 1] * X**deg
    targets = {}
    targets['lr_wpow'] = wpow
    targets['lr_wpowx'] = wpowx
    targets['lr_wxpow'] = wxpow
    if save:
        save_probetargets(targets, f'lr_cca_targets_deg{maxdeg}.pth', datatype, traintest)
    return targets

def generate_reverselr_targets(datatype = 'rlinreg1', traintest = 'train'):
    ccatargets = generate_lr_cca_targets(datatype, traintest, save = False)
    rlr_targets = {}
    rlr_targets['rlr_wi2'] = ccatargets['lr_wpow'][:, :, :2]
    # rlr_targets['rlr_wi1'] = ccatargets['lr_wpow'][:, :, :1]
    # rlr_targets['rlr_wix2'] = ccatargets['lr_wpowx'][:, :, :2]
    save_probetargets(rlr_targets, 'rlr_targets.pth', datatype, traintest)


def save_probetargets(targets, fname, datatype, traintest):
    bigdir = 'probe_targets'
    dir = f'{datatype}_{traintest}'
    if dir not in os.listdir(bigdir):
        os.mkdir(f'{bigdir}/{dir}')
    torch.save(targets, f'{bigdir}/{dir}/{fname}')

if __name__ == '__main__':


    for datatype in ['overdamped']:
        for traintest in ['train', 'test']:
            for reverse in [True, False]:
                print(f'Generating targets for {datatype} {traintest}')
                generate_exp_targets(datatype, traintest, maxdeg = 5, reverse = reverse)
                generate_rk_targets(datatype, traintest, maxdeg = 5, reverse = reverse)
                if reverse:
                    generate_rkexp_targets_REVERSE(datatype, traintest, maxdeg = 5)
