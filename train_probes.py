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
from generate_probe_targets import get_expanded_data


def create_probetarget_df(datatypes, traintests, reverse = False):
    prec = ''
    if reverse == True:
        prec = 'r'
    allmethods = {'underdamped': [f'{prec}rk', f'{prec}eA', f'{prec}eA_cca', f'{prec}rk_cca'],
                  'linreg1': ['lr'],
                  'linreg1cca': ['lr_cca'],
                  'rlinreg1': ['rlr'],
                  'wlinreg1cca': ['lr_cca']}
    allmethods['damped'] = allmethods['underdamped']
    allmethods['overdamped'] = allmethods['underdamped']
    allmethods['undamped'] = allmethods['underdamped']
    nodegmethods = ['lr', 'rlr']
    alldfs = None
    for datatype in datatypes:
        for traintest in traintests:
            probetargets = {'targetmethod':[], 'targetname':[], 'targetpath':[], 'deg': [],'datatype':[], 'traintest':[]}
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
            if alldfs is None:
                alldfs = df
            else:
                alldfs = pd.concat([alldfs, df])
    return alldfs


def create_probe_model_df(modeltypes, datatypes, traintests, reverse = False):
    # zip modeltypes and datatypes
    # get all combinations fo modeltypes and datatypes
    allmodeldatatraintest = [(modeltype, datatype, traintest) for modeltype in modeltypes for datatype in datatypes for traintest in traintests]
    alldfs = None
    for modeltype, datatype, traintest in allmodeldatatraintest:
        model_hs = pd.read_csv(f'dfs/{modeltype}_{datatype}_{traintest}_model_hss.csv', index_col = 0)
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
            df.to_csv(f'dfs/{modeltype}_{datatype}_{traintest}_reverseprobetorun.csv')
        else:
            df.to_csv(f'dfs/{modeltype}_{datatype}_{traintest}_probetorun.csv')
        if alldfs is None:
            alldfs = df
        else:
            alldfs = pd.concat([alldfs, df])
    return alldfs



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


def get_dftorun(modeltypes, datatypes, traintests, savename, reverse = False, cca = False):
    allmodeldatatraintest = [(modeltype, datatype, traintest) for modeltype in modeltypes for datatype in datatypes for traintest in traintests]
    df = None
    i = 0
    for modeltype, datatype, traintest in allmodeldatatraintest:
        print(f'{i}/{len(allmodeldatatraintest)}')
        if reverse:
            minidf = pd.read_csv(f'dfs/{modeltype}_{datatype}_{traintest}_reverseprobetorun.csv', index_col = 0)
        else:
            minidf = pd.read_csv(f'dfs/{modeltype}_{datatype}_{traintest}_probetorun.csv', index_col = 0)
        # concat to df or create df
        if df is None:
            df = minidf
        else:
            df = pd.concat([df, minidf])
        i+=1

    if not cca:
        df = df[~df['p-targetmethod'].str.contains('cca')]
    elif cca:
        df = df[df['p-targetmethod'].str.contains('cca')]
    # reset index of df
    df = df.reset_index(drop = True)
    # save df
    if reverse:
        df.to_csv(f'dfs/{savename}_reverseprobetorun.csv')
    else:
        df.to_csv(f'dfs/{savename}_probetorun.csv')

    return df

def train_probe(input, output, savepath, save = True):
    # assumes modelname doesnt have .pth
    input, output = input.detach().numpy(), output.detach().numpy()
    if len(input.shape) == 1:
        input = input.reshape(-1, 1)
    clf = Ridge(alpha=1.0)
    clf.fit(input, output)
    # save clf
    if save:
        torch.save(clf, savepath)
    r2 = clf.score(input, output)
    # get mse
    mse = ((output - clf.predict(input))**2).mean()
    return r2, mse


def train_probes(modeltypes, datatypes, traintests, savename, my_task_id =0,num_tasks = 1, reverse = False):
    
    # get rid of all entries in df with "cca" in their targetmethod
    df = get_dftorun(modeltypes, datatypes, traintests, savename, reverse = reverse, cca = False)
    print(len(df))

    savedir = f'proberesults_{savename}'
    if savedir not in os.listdir('dfs/proberesults'):
        os.mkdir(f'dfs/proberesults/{savedir}')
    
    if my_task_id is None:
        my_task_id = int(sys.argv[1])
    if num_tasks is None:
        num_tasks = int(sys.argv[2])

    indices = list(df.index)
    my_indices = indices[my_task_id:len(indices):num_tasks]
    minidf = {key:[] for key in df.columns}
    minidf['p-r2'] = []
    minidf['p-mse'] = []
    minidf['p-savepath'] = []
    print(my_indices[0], my_indices[-1])
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
        save = reverse #only reverse probes are worth saving
        if save:
            savepath = get_savepath(modelpath, target, layer, inlayerpos, pCL, append)
        else: savepath = '' # no reason to save
        r2, mse = train_probe(input,output, savepath, save = save)
        print(f'{index}: layer {layer}, inlayer {inlayerpos}, CL {pCL}|  {target}, R^2 = {r2:.3f}')
        for key in df.columns:
            minidf[key].append(row[key])
        minidf['p-r2'].append(r2)
        minidf['p-mse'].append(mse)
        minidf['p-savepath'].append(savepath)
        if (i+1) % 100 == 0:
            minidfdf = pd.DataFrame(minidf)
            minidfdf.to_csv(f'dfs/proberesults/{savedir}/proberesults_{savename}_{my_task_id}.csv')
    
    
    minidfdf = pd.DataFrame(minidf)
    minidfdf.to_csv(f'dfs/proberesults/{savedir}/proberesults_{savename}_{my_task_id}.csv')
    #minidf.to_csv(f'dfs/proberesults/proberesults_{my_task_id}.csv')

def train_cca_probe(input, output, savepath, save = True):
    input, output = input.detach().numpy(), output.detach().numpy()
    cca = CCA(n_components=1)
    cca.fit(input, output)
    if save:
        torch.save(cca, savepath)
    X_c, Y_c = cca.transform(input, output)
    canonical_correlations = np.corrcoef(X_c.T, Y_c.T).diagonal(offset=X_c.shape[1])
    r2 = canonical_correlations**2
    mse = mean_squared_error(X_c, Y_c)
    return r2[0], mse


def train_cca_probes(modeltypes, datatypes, traintests, savename, maxdeg = 5, my_task_id =0, num_tasks = 1, save = False):
    if my_task_id is None:
        my_task_id = int(sys.argv[1])
    if num_tasks is None:
        num_tasks = int(sys.argv[2])

    df = get_dftorun(modeltypes, datatypes, traintests, reverse = False, cca = True)
    print(len(df))

    # get all indices in df

    savedir = f'proberesults_{savename}'
    if savedir not in os.listdir('dfs/proberesults'):
        os.mkdir(f'dfs/proberesults/{savedir}')
    
    indices = list(df.index)
    my_indices = indices[my_task_id:len(indices):num_tasks]
    minidf = {key:[] for key in df.columns}
    minidf['cca-r2'] = []
    minidf['cca-deg'] = []
    minidf['cca-mse'] = []
    minidf['cca-savepath'] = []

    epoch_pbar = tqdm(range(len(my_indices)), desc='Training Progress')

    saveprobes = f'dfs/proberesults/{savedir}/proberesults_{savename}_{my_task_id}.csv'

    for i in epoch_pbar:
        index = my_indices[i]
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
            if save:
                savepath = get_savepath(modelpath, target, layer, inlayerpos, pCL, append = f'deg{deg}')
            else: savepath = ''
            varhs = torch.var(hs, dim=0, unbiased=False)
            print(max(varhs), varhs.mean())
            if max(varhs) < 1e-4:
                r2, mse = 0, float('inf')
            else:
                r2, mse = train_cca_probe(hs,targetval, savepath, save)
            #print(f'{index}: layer {layer}, inlayer {inlayerpos}, CL {pCL}|  {target}, deg{deg}, R^2 = {r2:.3f}')
            for key in df.columns:
                minidf[key].append(row[key])
            minidf['cca-r2'].append(r2)
            minidf['cca-mse'].append(mse)
            minidf['cca-deg'].append(deg)
            minidf['cca-savepath'].append(savepath)

            epoch_pbar.set_description(f'Epoch {i + 1}/{len(my_indices)}')
            epoch_pbar.set_postfix({'layer': f'{layer}',
                        'inlayer': f'{inlayerpos}',
                        'CL': f'{pCL}',
                        'target': f'{target}',
                        'deg': f'{deg}',
                        'R^2': f'{r2:.3f}'
                        })
            if (i+1) % 1000 == 0:
                minidfdf = pd.DataFrame(minidf)
                minidfdf.to_csv(saveprobes)

    
    minidfdf = pd.DataFrame(minidf)
    minidfdf.to_csv(saveprobes)
    



if __name__ == '__main__':

    my_task_id, num_tasks = 0,48

    modeltypes = ['undamped']
    datatypes = ['undamped']
    traintests = ['train', 'test']

    # pdf = create_probetarget_df(datatypes, traintests, reverse = True)
    # print(pdf)
    # # print(len(pdf))
    # mpdf = create_probe_model_df(modeltypes, datatypes, traintests, reverse = True)
    # print(len(mpdf))
    savestr = 'ALLUNDAMPEDSPRING'

    # ccastr = savestr + 'CCA'
    train_probes(modeltypes, datatypes, traintests, savestr, my_task_id, num_tasks, reverse = True)
    #train_probes(modeltypes, datatypes, traintests, savestr, my_task_id, num_tasks, reverse = False)
    # maxdeg = 5
    #train_cca_probes(modeltypes, datatypes, traintests, ccastr, maxdeg, my_task_id, num_tasks)
    #create_probe_model_df(modeltypes, datatypes, traintests)

    #generate_exp_targets(datatype, traintest)
    #generate_exp_targets(datatype, traintest, reverse=True)
    # pdf = create_probetarget_df(datatype, traintest, reverse = True)
    # print(len(pdf))
    # mpdf = create_probe_model_df(datatype, traintest, reverse = True)
    # print(len(mpdf))
    #train_probes(datatype, traintest, my_task_id, num_tasks, reverse = True)
    
    
    #create_probetarget_df(datatype, traintest)
    # pdf = create_probetarget_df(datatype, traintest)
    # pmdf = create_probe_model_df(datatype, traintest)
    # generate_lr_cca_targets(datatype, traintest)
    # get_model_hs_df(lrdf, datatype, traintest)
    # create_probetarget_df(datatype, traintest)
    # mpdf = create_probe_model_df(datatype, traintest)
    
    # mpdf = mpdf[mpdf['p-targetname'] == 'lr_wpow']
    # # reset index
    # mpdf = mpdf.reset_index(drop = True)
    # datatype = 'wlinreg1cca'
    # mpdf.to_csv(f'dfs/{datatype}_{traintest}_probetorun.csv')
    # print(mpdf['m-layer'].unique(), mpdf['m-emb'].unique())
    #train_cca_probes(datatype, traintest, 5, my_task_id, num_tasks)

    # datatype, traintest = 'rlinreg1', 'train'
    # get_model_hs_df(lrdf, datatype, traintest)
    # generate_reverselr_targets(datatype, traintest)
    # create_probetarget_df(datatype, traintest, save = True, reverse = True)
    # create_probe_model_df(datatype, traintest, reverse = True)

    #train_probes(datatype, traintest, my_task_id, num_tasks, reverse = False)





