import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import re
from util import load_model, get_data, get_log_log_linear

def get_df_models():
    # NEED LOSS PATHS SOMEHOW

    models = {
        "datatype": [],
        "emb": [],
        "layer": [],
        "epoch": [],
        "CL": [],
        "lr": [],
        "totalepochs": [],
        "batch": [],
        'modelpath': []
    }
    for root, dirs, files in os.walk('models'):
        for file in files:
            filename = os.path.join(root, file)
            pattern = r"(\w+)_(\d+)emb_(\d+)layer_(\d+)CL_(\d+)epochs_([\d\.]+)lr_(\d+)batch_model(?:_epoch(\d+))?\.pth"
            match = re.search(pattern, filename)
            if match:
                models['datatype'].append(match.group(1))
                models['emb'].append(int(match.group(2)))
                models['layer'].append(int(match.group(3)))
                models['CL'].append(int(match.group(4)))
                models['totalepochs'].append(int(match.group(5)))
                models['lr'].append(float(match.group(6)))
                models['batch'].append(int(match.group(7)))
                epoch = match.group(8) if match.group(8) else match.group(5)
                models['epoch'].append(int(epoch))
                models['modelpath'].append(filename)
            else:
                continue
                #print("No match found in file:", filename)
    df = pd.DataFrame(models)
    # sort df by layers and emb
    df = df.sort_values(by=['layer', 'emb'])
    return df

def plot_ICL(modeldf, datatype = 'underdamped', traintest = 'train'):
    plt.figure(figsize = (10,10))
    data = {}
    for index, row in modeldf.iterrows():
        model = load_model(row, datatype)
        if 'linreg' in datatype:
            _, sequences = get_data(datatype, traintest)
            X, y = sequences[:, :-1], sequences[:, 1:]
            print(X.shape, 'X here')
            ypred = model(X)
            y, ypred = y[:, 0::2], ypred[:, 0::2]
        else:
            _, _, sequences, _, _ = get_data(datatype, traintest)
            X, y = sequences[:, :-1], sequences[:, 1:]
            print(X.shape, 'X here')
            ypred = model(X)
        CLs = range(1, y.shape[1]+1)
        mses = ((ypred - y)**2).mean(dim=2).detach()
        mses = mses.mean(dim=0).numpy()
        #print(mses.shape)
        #mses = [mses[:,:i].mean() for i in CLs]
        slope, intercept, r_value = get_log_log_linear(CLs, mses)
        plt.plot(CLs, mses, label = f'{row["emb"]}emb_{row["layer"]}layer Last MSE: {mses[-1]:.2e}')#log(MSE) = {slope:.4f}log(CL) + {intercept:.2f}, R^2 = {r_value**2:.2f}')
        data[(row['emb'], row['layer'])] = mses
    plt.xlabel("CL")
    plt.ylabel("MSE")
    plt.yscale('log')
    #plt.xscale('log')
    plt.title(f'MSE vs Context Length for {datatype} {traintest} data')
    plt.legend()
    plt.show()
    return data, CLs

def get_model_hs_df(modeldf, datatype = 'underdamped', traintest = 'train'):
    #gets hidden states for all models in modeldf, saves them, and returns a dataframe
    # with full data
    # saves hidden states for model with model. need to index layer and in layer position still. 
    hsdf  = {f'm-{key}':[] for key in modeldf.columns}
    hsdf['h-hspath'] = []
    hsdf['h-layerpos'] = []
    hsdf['h-inlayerpos'] = []
    hsdf['h-datatype'] = []
    hsdf['h-traintest'] = []
    for index, row in modeldf.iterrows():
        print(row['modelpath'])
        model = load_model(row, datatype)
        if 'linreg1' in datatype:
            _, sequences = get_data(datatype, traintest)
        else:
            _, _, sequences, _, _ = get_data(datatype, traintest)
        X, _ = sequences[:, :-1], sequences[:, 1:]
        _, hs = model.forward_hs(X)
        breakpoint()
        savepath = row['modelpath'][:-4]+'_hss.pth'
        for layer in hs:
            for inlayerpos in hs[layer]:
                hsdf['h-hspath'].append(savepath)
                hsdf['h-layerpos'].append(layer)
                hsdf['h-inlayerpos'].append(inlayerpos)
                hsdf['h-datatype'].append(datatype)
                hsdf['h-traintest'].append(traintest)
                for key in modeldf.columns:
                    hsdf[f'm-{key}'].append(row[key])
        torch.save(hs, savepath)
    hsdf = pd.DataFrame(hsdf)
    hsdf.to_csv(f'dfs/{datatype}_{traintest}_model_hss.csv')
    return hsdf
        
if __name__ == '__main__':
    df = get_df_models()
    df = df[df['epoch'] == 20000]
    df = df[df['datatype'] == 'linreg1']
    # emb not 16
    df = df[df['emb'] != 64]
    df = df[(df['emb'] == 32) & (df['layer'] == 5)]
    print(df)
    get_model_hs_df(df, datatype = 'linreg1', traintest = 'train')
    # df = df[df['emb'] == 16]
    # df = df[df['layer'] == 2]
    #plot_ICL(df, datatype = 'linreg1', traintest = 'test')
    #get_model_hs_df(df, datatype = 'linreg1cca', traintest = 'train')
    #get_model_hs_df(df)
    #plot_ICL(df, datatype = 'overdamped', traintest = 'test')