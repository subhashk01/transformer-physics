from util import get_data
import torch
from math import factorial
import os
import pandas as pd
from sklearn.linear_model import Ridge
import time
import sys

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

def save_probetargets(targets, fname, datatype, traintest):
    bigdir = 'probe_targets'
    dir = f'{datatype}_{traintest}'
    if dir not in os.listdir(bigdir):
        os.mkdir(f'{bigdir}/{dir}')
    torch.save(targets, f'{bigdir}/{dir}/{fname}')

def create_probetarget_df(datatype, traintest):
    methods = ['mw', 'rk', 'lm']
    probetargets = {'targetmethod':[], 'targetname':[], 'targetpath':[], 'deg': [],'datatype':[], 'traintest':[]}
    for method in methods:
        dir = f'probe_targets/{datatype}_{traintest}'
        fname = f'{method}_targets'
        if not method == 'mw':
            fname += '_deg5.pth'
        else: fname+='.pth'
        filepath = f'{dir}/{fname}'
        targets = torch.load(filepath)
        for key in targets:
            probetargets['targetmethod'].append(method)
            probetargets['targetname'].append(key)
            probetargets['targetpath'].append(filepath)
            if not method == 'mw':
                probetargets['deg'].append(key[-1])
            else: probetargets['deg'].append(None)
            probetargets['datatype'].append(datatype)
            probetargets['traintest'].append(traintest)
    df = pd.DataFrame(probetargets)
    # save df
    df.to_csv(f'dfs/{datatype}_{traintest}_probetargets.csv')

def create_probe_model_df(datatype, traintest):
    model_hs = pd.read_csv(f'dfs/{datatype}_{traintest}_model_hss.csv', index_col = 0)
    probetargets = pd.read_csv(f'dfs/{datatype}_{traintest}_probetargets.csv', index_col = 0)
    keys = [mkey for mkey in model_hs.columns]
    for pkey in probetargets.columns:
        keys.append(f'p-{pkey}')
    keys.append('p-CL')
    df = {key:[] for key in keys}
    omegas, gammas, deltat, X, y, x, v = get_expanded_data(datatype, traintest)
    for mindex, mrow in model_hs.iterrows():
        for pindex, prow in probetargets.iterrows():
            print(mindex, pindex)
            for CL in range(X.shape[1]):
                for mkey in model_hs.columns:
                    df[mkey].append(mrow[mkey])
                for pkey in probetargets.columns:
                    df[f'p-{pkey}'].append(prow[pkey])
                df['p-CL'].append(CL)
    df = pd.DataFrame(df)
    # save df
    df.to_csv(f'dfs/{datatype}_{traintest}_probetorun.csv')



def train_probe(input, output, modelpath, targetname, CL):
    # assumes modelname doesnt have .pth
    input, output = input.detach().numpy(), output.detach().numpy()
    modelname = modelpath[modelpath.rfind('/')+1:-4]
    mdir = f'probes/{modelname}'
    if modelname not in os.listdir('probes'):
        os.mkdir(mdir)
    totaldir = f'{mdir}/{targetname}'
    if targetname not in os.listdir(mdir):
        os.mkdir(totaldir)
    savepath = totaldir+'/' + f'{CL}.pth'
    clf = Ridge(alpha=1.0)
    clf.fit(input, output)
    # save clf
    torch.save(clf, savepath)
    r2 = clf.score(input, output)
    # get mse
    mse = ((output - clf.predict(input))**2).mean()
    return r2, mse, savepath


def train_probes(my_task_id =1,num_tasks = 1):
    if my_task_id is None:
        my_task_id = int(sys.argv[1])
    if num_tasks is None:
        num_tasks = int(sys.argv[2])
    

    #my_fnames = fnames[my_task_id:len(fnames):num_tasks]
    df = pd.read_csv('dfs/underdamped_train_probetorun.csv', index_col = 0)
    # get all indices in df 
    indices = list(df.index)
    my_indices = indices[my_task_id:len(indices):num_tasks]
    minidf = {key:[] for key in df.columns}
    minidf['p-r2'] = []
    minidf['p-mse'] = []
    minidf['p-savepath'] = []

    for i, index in enumerate(my_indices):
        print(index)
        row = df.iloc[index]
        pCL = row['p-CL']
        layer = row['h-layerpos']
        inlayerpos = row['h-inlayerpos']
        hss = torch.load(row['h-hspath'])
        hs = hss[layer][inlayerpos][:, pCL]
        target = row['p-targetname']
        targetpath = row['p-targetpath']
        targetval = torch.load(targetpath)[target][:, pCL]
        modelpath = row['m-modelpath']
        r2, mse, savepath = train_probe(hs,targetval, modelpath, target, pCL)
        for key in df.columns:
            minidf[key].append(row[key])
        minidf['p-r2'].append(r2)
        minidf['p-mse'].append(mse)
        minidf['p-savepath'].append(savepath)
        if i % 1000 == 0:
            minidfdf = pd.DataFrame(minidf)
            minidfdf.to_csv(f'dfs/proberesults_{my_task_id}.csv')
    minidfdf.to_csv(f'dfs/proberesults_{my_task_id}.csv')


    
    



if __name__ == '__main__':
    #generate_rk_targets('underdamped', 'train')
    #generate_mw_targets('underdamped', 'train')
    #create_probetarget_df('underdamped', 'train')
    #create_probe_model_df('underdamped', 'train')
    # df = pd.read_csv('dfs/underdamped_train_probetorun.csv')
    # row = df.iloc[123123]
    # pCL = row['p-CL']
    # emb = row['m-emb']
    # layer = row['h-layerpos']
    # inlayerpos = row['h-inlayerpos']
    # hss = torch.load(row['h-hspath'])
    # hs = hss[layer][inlayerpos][:, pCL]
    # target = row['p-targetname']
    # targetpath = row['p-targetpath']
    # targetval = torch.load(targetpath)[target][:, pCL]
    # #find index of last backslash in modelname
    # modelpath = row['m-modelpath']

    # train_probe(hs,targetval, modelpath, target, pCL)
    datatype, traintest = 'underdamped', 'train'
    generate_rk_targets(datatype, traintest, maxdeg = 5)
    generate_mw_targets(datatype, traintest)
    generate_lm_targets(datatype, traintest, maxdeg = 5)
    create_probe_model_df(datatype, traintest)
    train_probes(my_task_id =None,num_tasks = None)

