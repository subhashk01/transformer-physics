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
from analyze_models import get_model_hs_df, get_model_df

def get_expanded_data(datatype, traintest):
    gammas, omegas, sequences, times, deltat = get_data(datatype, traintest)
    # tiles omegas and gammas to match the shape of sequences
    X, y = sequences[:,:-1], sequences[:,1:]
    x, v = X[:,:,0], X[:,:,1]
    omegas = omegas.repeat(X.shape[1], 1).T
    gammas = gammas.repeat(X.shape[1], 1).T
    deltat = deltat.repeat(X.shape[1], 1).T
    return omegas, gammas, deltat, X, y, x, v 

def generate_rk_targets(datatype, traintest, maxdeg = 5):
    criterion = torch.nn.MSELoss()
    omegas, gammas, deltat, X, y, x, v = get_expanded_data(datatype, traintest)
    
    A = torch.zeros((X.shape[0], X.shape[1],2,2))
    A[:, :, 0, 1] = 1
    A[:, :, 1, 0] = -omegas**2
    A[:, :, 1, 1] = -2*gammas
    targets = {}
    # 5000 x 65 x 2 x 2 identity matrix
    totalmat = torch.eye(2).repeat(X.shape[0], X.shape[1], 1, 1)
    for i in range(1,maxdeg+1):
        Ai = torch.matrix_power(A, i)
        targets[f'rk_A10_j{i}'] = Ai[:,:,1,0]
        targets[f'rk_A11_j{i}'] = Ai[:,:,1,1]
        targets[f'rk_dt_j{i}'] = deltat**i
        targets[f'rk_A10dt_j{i}'] = Ai[:,:,1,0] * deltat**i
        targets[f'rk_A11dt_j{i}'] = Ai[:,:,1,1] * deltat**i
        targets[f'rk_A10dtx_j{i}'] = Ai[:,:,1,0] * deltat**i * x
        targets[f'rk_A11dtv_j{i}'] = Ai[:,:,1,1] * deltat**i * v

        # make prediction for each degree
        coef = 1/factorial(i)
        currentmat = coef * Ai * (deltat**i).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, 2)
        totalmat += currentmat 
        totalmat = totalmat.squeeze(-1)
        ypred = (totalmat @ X.unsqueeze(-1)).squeeze(-1)
        loss = criterion(ypred, y)
        print(f"Loss for degree {i}: {loss:.2e}")
    
    fname = f'rk_targets_deg{maxdeg}.pth'
    save_probetargets(targets, fname, datatype, traintest)

def generate_mw_targets(datatype, traintest, maxdeg = 5):
    criterion = torch.nn.MSELoss()
    omega0, gammas, deltat, X, y, x, v = get_expanded_data(datatype, traintest)
    omegas = torch.sqrt(omega0**2 - gammas**2)
    beta = -(gammas**2 + omegas**2)/omegas
    targets = {}
    targets['mw_w'] = omegas
    targets['mw_b'] = beta
    targets['mw_coswt'] = torch.cos(omegas * deltat)
    targets['mw_sinwt'] = torch.sin(omegas * deltat)
    targets['mw_gwsinwt'] = gammas/omegas * torch.sin(omegas * deltat)
    targets['mw_w00'] = targets['mw_coswt']+targets['mw_gwsinwt']
    targets['mw_w01'] = 1/omegas * targets['mw_sinwt']
    targets['mw_w10'] = beta * torch.sin(omegas * deltat)
    targets['mw_w11'] = targets['mw_coswt'] - targets['mw_gwsinwt']

    targets['mw_w00x'] = targets['mw_w00'] * x
    targets['mw_w01v'] = targets['mw_w01'] * v
    targets['mw_w10x'] = targets['mw_w10'] * x
    targets['mw_w11v'] = targets['mw_w11'] * v
    
    egt_dict = {}
    for key in targets:
        # extract post underscore
        post_underscore = key.split('_')[-1]
        egt_dict[f'mw_egt{post_underscore}'] = targets[key] * torch.exp(-gammas*deltat)
    targets.update(egt_dict)
    targets['mw_egt'] = torch.exp(-gammas*deltat)

    #mw prediction
    ypred = torch.zeros(y.shape)
    ypred[:,:,0] = targets['mw_egtw00x'] + targets['mw_egtw01v']
    ypred[:,:,1] = targets['mw_egtw10x'] + targets['mw_egtw11v']
    mse = criterion(ypred, y)
    print(mse)
    save_probetargets(targets, f'mw_targets.pth', datatype, traintest)


def generate_exp_targets(datatype, traintest):
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
    def compare_exp_mw():
        mw = torch.load(f'probe_targets/{datatype}_{traintest}/mw_targets.pth')
        mw_weights = torch.zeros(eAdt.shape)
        mw_weights[:,:, 0, 0] = mw['mw_egtw00']
        mw_weights[:, :, 0, 1] = mw['mw_egtw01']
        mw_weights[:, :, 1, 0] = mw['mw_egtw10']
        mw_weights[:, :, 1, 1] = mw['mw_egtw11']
        print(mw_weights[1000,0])
    targets = {}
    

    targets['eAdt00'] = eAdt[:, :, 0, 0]
    targets['eAdt01'] = eAdt[:, :, 0, 1]
    targets['eAdt10'] = eAdt[:, :, 1, 0]
    targets['eAdt11'] = eAdt[:, :, 1, 1]
    targets['eAdt00x'] = targets['eAdt00'] * x
    targets['eAdt01v'] = targets['eAdt01'] * v
    targets['eAdt10x'] = targets['eAdt10'] * x
    targets['eAdt11v'] = targets['eAdt11'] * v
    test_exp()
    compare_exp_mw()
    save_probetargets(targets, f'eA_targets.pth', datatype, traintest)



def generate_lm_targets(datatype, traintest, maxdeg = 5):
    criterion = torch.nn.MSELoss()
    omegas, gammas, deltat, X, y, x, v = get_expanded_data(datatype, traintest)
    buffer = torch.zeros(X.shape[0], maxdeg - 1, X.shape[2])
    # concat X and buffer
    X_buf = torch.concat((buffer, X), dim = 1)
    x_buf, v_buf = X_buf[:,:,0], X_buf[:,:,1]
    targets = {}
    print(x_buf[0])
    for deg in range(1, maxdeg+1):
        start = maxdeg - deg
        end = start + X.shape[1]
        print(deg, start, end)
        targets[f'lm_x{deg}'] = x_buf[:, start : end]
        targets[f'lm_v{deg}'] = v_buf[:, start : end]
        targets[f'lm_w2x{deg}'] = omegas**2 * targets[f'lm_x{deg}']
        targets[f'lm_gv{deg}'] = gammas * targets[f'lm_v{deg}']
        targets[f'lm_dtw2x{deg}'] = deltat * targets[f'lm_w2x{deg}']
        targets[f'lm_dtgv{deg}'] = deltat * targets[f'lm_gv{deg}'] 
        targets[f'lm_dtv{deg}'] = deltat * targets[f'lm_v{deg}']
    targets['lm_x0'] = y[:,:,0]
    targets['lm_v0'] = y[:,:,1]
    targets['lm_w2x0'] = omegas**2 * y[:,:,0]
    targets['lm_gv0'] = gammas * y[:,:,1]

    def lm_prediction(targets):
        xpred = targets[f'lm_x1']
        vpred = targets[f'lm_v1']
        print(xpred == x)
        coef = [23/12,-16/12, 5/12]
        for deg in range(1, 4):
            xpred += coef[deg-1] * targets[f'lm_dtv{deg}']
            vpred += coef[deg-1] * (-1 * targets[f'lm_dtw2x{deg}'] + -2 * targets[f'lm_dtgv{deg}'])
        pred = torch.zeros(y.shape)
        pred[:,:,0] = xpred
        pred[:,:,1] = vpred
        mse = criterion(pred[:, maxdeg:], y[:, maxdeg:])
        print(mse)
    lm_prediction(targets)

    fname = f'lm_targets_deg{maxdeg}.pth'
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

def create_probetarget_df(datatype, traintest, save = True, reverse = False):
    allmethods = {'underdamped': ['mw', 'rk', 'lm', 'eA'],
                  'linreg1': ['lr'],
                  'linreg1cca': ['lr_cca'],
                  'rlinreg1': ['rlr'],
                  'wlinreg1cca': ['lr_cca']}
    probetargets = {'targetmethod':[], 'targetname':[], 'targetpath':[], 'deg': [],'datatype':[], 'traintest':[]}
    nodegmethods = ['mw', 'eA', 'lr', 'rlr']
    for method in allmethods[datatype]:
        dir = f'probe_targets/{datatype}_{traintest}'
        fname = f'{method}_targets'
        if not method in nodegmethods:
            fname += '_deg5.pth'
        else: fname+='.pth'
        filepath = f'{dir}/{fname}'
        targets = torch.load(filepath)
        for key in targets:
            probetargets['targetmethod'].append(method)
            probetargets['targetname'].append(key)
            probetargets['targetpath'].append(filepath)
            if not method in nodegmethods or method=='rlr':
                probetargets['deg'].append(key[-1])
            else: probetargets['deg'].append(None)
            probetargets['datatype'].append(datatype)
            probetargets['traintest'].append(traintest)
    df = pd.DataFrame(probetargets)
    # save df
    if reverse:
        df.to_csv(f'dfs/{datatype}_{traintest}_reverseprobetargets.csv')
    else:
        df.to_csv(f'dfs/{datatype}_{traintest}_probetargets.csv')
    return df


def create_probe_model_df(datatype, traintest, reverse = False):
    model_hs = pd.read_csv(f'dfs/{datatype}_{traintest}_model_hss.csv', index_col = 0)
    if reverse:
        probetargets = pd.read_csv(f'dfs/{datatype}_{traintest}_reverseprobetargets.csv', index_col = 0)
    else:
        probetargets = pd.read_csv(f'dfs/{datatype}_{traintest}_probetargets.csv', index_col = 0)
    keys = [mkey for mkey in model_hs.columns]
    for pkey in probetargets.columns:
        keys.append(f'p-{pkey}')
    keys.append('p-CL')
    df = {key:[] for key in keys}
    if 'linreg' in datatype:
        _, sequences = get_data(datatype, traintest)
        X, y = sequences[:, :-1], sequences[:, 1:]
    else: # spring data
        omegas, gammas, deltat, X, y, x, v = get_expanded_data(datatype, traintest)
    for mindex, mrow in model_hs.iterrows():
        for pindex, prow in probetargets.iterrows():
            print(mindex, pindex)
            for CL in range(X.shape[1]):
                if 'linreg' in datatype:
                    if CL % 2 == 1:
                        continue # only want even CLs, where predictions happen
                for mkey in model_hs.columns:
                    df[mkey].append(mrow[mkey])
                for pkey in probetargets.columns:
                    df[f'p-{pkey}'].append(prow[pkey])
                df['p-CL'].append(CL)
    df = pd.DataFrame(df)
    if reverse:
        df.to_csv(f'dfs/{datatype}_{traintest}_reverseprobetorun.csv')
    else:
        df.to_csv(f'dfs/{datatype}_{traintest}_probetorun.csv')
    return df


def get_savepath(modelpath, targetname, layer, inlayerpos, CL, append = ''):
    modelname = modelpath[modelpath.rfind('/')+1:-4]
    mdir = f'probes/{modelname}'
    if modelname not in os.listdir('probes'):
        os.mkdir(mdir)
    totaldir = f'{mdir}/{targetname}'
    if targetname not in os.listdir(mdir):
        os.mkdir(totaldir)
    fname = f'{targetname}_{layer}layer_{inlayerpos}_{CL}CL'
    if len(append):
        fname = f'{fname}_{append}'
    savepath = f'{totaldir}/{fname}.pth'
    return savepath


def train_probe(input, output, savepath):
    # assumes modelname doesnt have .pth
    input, output = input.detach().numpy(), output.detach().numpy()
    clf = Ridge(alpha=1.0)
    clf.fit(input, output)
    # save clf
    torch.save(clf, savepath)
    r2 = clf.score(input, output)
    # get mse
    mse = ((output - clf.predict(input))**2).mean()
    return r2, mse


def train_probes(datatype, traintest, my_task_id =0,num_tasks = 1, reverse = False):
    if my_task_id is None:
        my_task_id = int(sys.argv[1])
    if num_tasks is None:
        num_tasks = int(sys.argv[2])
    

    #my_fnames = fnames[my_task_id:len(fnames):num_tasks]
    if reverse:
        df = pd.read_csv(f'dfs/{datatype}_{traintest}_reverseprobetorun.csv', index_col = 0)
        savedir = f'reverseproberesults_{datatype}_{traintest}'
    else:
        df = pd.read_csv(f'dfs/{datatype}_{traintest}_probetorun.csv', index_col = 0)
        savedir = f'proberesults_{datatype}_{traintest}'
    # get all indices in df
    print(len(df))

    
    if savedir not in os.listdir('dfs/proberesults'):
        os.mkdir(f'dfs/proberesults/{savedir}')
    
    indices = list(df.index)
    my_indices = indices[my_task_id:len(indices):num_tasks]
    minidf = {key:[] for key in df.columns}
    minidf['p-r2'] = []
    minidf['p-mse'] = []
    minidf['p-savepath'] = []
    print(max(my_indices))
    for i, index in enumerate(my_indices):
        row = df.iloc[index]
        pCL = row['p-CL']
        layer = row['h-layerpos']
        inlayerpos = row['h-inlayerpos']
        hpath = row['h-hspath']
        hss = torch.load(hpath)
        hs = hss[layer][inlayerpos][:, pCL]
        target = row['p-targetname']
        targetpath = row['p-targetpath']
        targetval = torch.load(targetpath)[target][:, pCL]
        modelpath = row['m-modelpath']
        if reverse:
            append = 'reverse'
            input, output = targetval, hs
        else:
            append = ''
            input, output = hs, targetval
        savepath = get_savepath(modelpath, target, layer, inlayerpos, pCL, append)
        r2, mse = train_probe(input,output, savepath)
        print(f'{index}: layer {layer}, inlayer {inlayerpos}, CL {pCL}|  {target}, R^2 = {r2:.3f}')
        for key in df.columns:
            minidf[key].append(row[key])
        minidf['p-r2'].append(r2)
        minidf['p-mse'].append(mse)
        minidf['p-savepath'].append(savepath)
    
    
    minidfdf = pd.DataFrame(minidf)
    minidfdf.to_csv(f'dfs/proberesults/{savedir}/proberesults_{datatype}_{traintest}_{my_task_id}.csv')
    #minidf.to_csv(f'dfs/proberesults/proberesults_{my_task_id}.csv')

def train_cca_probe(input, output, savepath):
    input, output = input.detach().numpy(), output.detach().numpy()
    cca = CCA(n_components=1)
    cca.fit(input, output)
    torch.save(cca, savepath)
    X_c, Y_c = cca.transform(input, output)
    canonical_correlations = np.corrcoef(X_c.T, Y_c.T).diagonal(offset=X_c.shape[1])
    r2 = canonical_correlations**2
    mse = mean_squared_error(X_c, Y_c)
    return r2[0], mse


def train_cca_probes(datatype, traintest, maxdeg, my_task_id =0,num_tasks = 1):
    if my_task_id is None:
        my_task_id = int(sys.argv[1])
    if num_tasks is None:
        num_tasks = int(sys.argv[2])

    df = pd.read_csv(f'dfs/{datatype}_{traintest}_probetorun.csv', index_col = 0)
    # get all indices in df

    savedir = f'proberesults_{datatype}_{traintest}'
    if savedir not in os.listdir('dfs/proberesults'):
        os.mkdir(f'dfs/proberesults/{savedir}')
    
    indices = list(df.index)
    my_indices = indices[my_task_id:len(indices):num_tasks]
    minidf = {key:[] for key in df.columns}
    minidf['cca-r2'] = []
    minidf['cca-deg'] = []
    minidf['cca-mse'] = []
    minidf['cca-savepath'] = []
    
    for i, index in enumerate(my_indices):
        for deg in range(1, maxdeg+1):
            row = df.iloc[index]
            pCL = row['p-CL']
            layer = row['h-layerpos']
            inlayerpos = row['h-inlayerpos']
            hss = torch.load(row['h-hspath'])
            hs = hss[layer][inlayerpos][:, pCL]
            target = row['p-targetname']
            targetpath = row['p-targetpath']
            targetval = torch.load(targetpath)[target][:, pCL, :deg]
            modelpath = row['m-modelpath']
            savepath = get_savepath(modelpath, target, layer, inlayerpos, pCL, append = f'deg{deg}')
            r2, mse = train_cca_probe(hs,targetval, savepath)
            print(f'{index}: layer {layer}, inlayer {inlayerpos}, CL {pCL}|  {target}, deg{deg}, R^2 = {r2:.3f}')
            for key in df.columns:
                minidf[key].append(row[key])
            minidf['cca-r2'].append(r2)
            minidf['cca-mse'].append(mse)
            minidf['cca-deg'].append(deg)
            minidf['cca-savepath'].append(savepath)
            # if (i+1) % 100 == 0:
            #     minidfdf = pd.DataFrame(minidf)
            #     minidfdf.to_csv(f'dfs/proberesults/{savedir}/proberesults_{datatype}_{traintest}_{my_task_id}.csv')

    
    minidfdf = pd.DataFrame(minidf)
    minidfdf.to_csv(f'dfs/proberesults/{savedir}/proberesults_{datatype}_{traintest}_{my_task_id}.csv')
    



if __name__ == '__main__':

    df = get_model_df()
    df = df[df['epoch'] == 20000]
    lrdf = df[df['datatype'] == 'linreg1']
    lrdf = lrdf[lrdf['emb'] != 64]
    my_task_id, num_tasks = 0,1


    datatype, traintest = 'wlinreg1cca', 'train'
    generate_lr_cca_targets(datatype, traintest)
    get_model_hs_df(lrdf, datatype, traintest)
    create_probetarget_df(datatype, traintest)
    mpdf = create_probe_model_df(datatype, traintest)
    
    mpdf = mpdf[mpdf['p-targetname'] == 'lr_wpow']
    # reset index
    mpdf = mpdf.reset_index(drop = True)
    datatype = 'wlinreg1cca'
    mpdf.to_csv(f'dfs/{datatype}_{traintest}_probetorun.csv')
    print(mpdf['m-layer'].unique(), mpdf['m-emb'].unique())
    #train_cca_probes(datatype, traintest, 5, my_task_id, num_tasks)

    datatype, traintest = 'rlinreg1', 'train'
    get_model_hs_df(lrdf, datatype, traintest)
    generate_reverselr_targets(datatype, traintest)
    create_probetarget_df(datatype, traintest, save = True, reverse = True)
    create_probe_model_df(datatype, traintest, reverse = True)

    #train_probes(datatype, traintest, my_task_id, num_tasks, reverse = True)





