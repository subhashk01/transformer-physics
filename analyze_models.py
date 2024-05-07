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
        "modeltype": [],
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
                models['modeltype'].append(match.group(1))
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

def plot_ICL(modeldf, datatype = 'underdamped', traintest = 'train', modeltype = ''):
    plt.figure(figsize = (10,10))
    savedata = {'emb':[], 'layer':[], 'CL':[], 'mse':[]}
    if len(modeltype) > 0:
        savepath = f'dfs/{datatype}_{traintest}_{modeltype}_ICL.csv'
    else: savepath = f'dfs/{datatype}_{traintest}_ICL.csv'
    for index, row in modeldf.iterrows():
        model = load_model(row, datatype)
        if 'linreg' in datatype:
            _, sequences = get_data(datatype, traintest)
            X, y = sequences[:, :-1], sequences[:, 1:]
            ypred = model(X)
            y, ypred = y[:, 0::2], ypred[:, 0::2]
        else:
            _, _, sequences, _, _ = get_data(datatype, traintest)
            X, y = sequences[:, :-1], sequences[:, 1:]
            ypred = model(X)
        CLs = range(1, y.shape[1]+1)
        mses = ((ypred - y)**2).mean(dim=2).detach()
        mses = mses.mean(dim=0).numpy()
        for i in range(len(CLs)):
            savedata['emb'].append(row['emb'])
            savedata['layer'].append(row['layer'])
            CL = i
            if 'linreg' in datatype:
                CL = i*2
            savedata['CL'].append(CL)
            savedata['mse'].append(mses[i])
        #print(mses.shape)
        #mses = [mses[:,:i].mean() for i in CLs]
        slope, intercept, r_value = get_log_log_linear(CLs, mses)
        label = f'{row["emb"]}emb_{row["layer"]}layer Last MSE: {mses[-1]:.2e}'
        #print(label)
        plt.plot(CLs, mses, label = label)#log(MSE) = {slope:.4f}log(CL) + {intercept:.2f}, R^2 = {r_value**2:.2f}')
        df = pd.DataFrame(savedata)
        df.to_csv(savepath)
    # plt.xlabel("CL")
    # plt.ylabel("MSE")
    # plt.yscale('log')
    # #plt.xscale('log')
    # plt.title(f'MSE vs Context Length for {datatype} {traintest} data {modeltype}')
    # plt.legend(loc = 'upper right')
    # plt.show()

def plot_ICL_damped():
    plt.rcParams.update({'font.size': 6})
    fig, axs = plt.subplots(1,2, figsize = (6,3), sharey = True)
    plt.subplots_adjust(wspace = 0)
    for i, datatype in enumerate(['underdamped', 'overdamped']):
        colors = 'rbg'
        ax = axs[i]
        for j, modeltype in enumerate(['underdamped', 'overdamped', 'damped']):
            modeldf = pd.read_csv(f'dfs/{datatype}_test_{modeltype}_ICL.csv')
            #mdflast = modeldf[modeldf['CL'] == modeldf['CL'].max()]
            #make model_best the model with best average mse across CL
            model_best = modeldf.groupby(['emb', 'layer']).mean().reset_index()
            model_best = model_best[model_best['mse'] == model_best['mse'].min()]
            #model_best = mdflast[mdflast['mse'] == mdflast['mse'].min()]
            emb, layer = model_best['emb'].values[0], model_best['layer'].values[0]
            allloss_modelbest = modeldf[(modeldf['emb'] == emb) & (modeldf['layer'] == layer)]
            allloss_modelbest = allloss_modelbest.sort_values(by = 'CL')
            CLs = allloss_modelbest['CL']
            mses = allloss_modelbest['mse']
            ax.plot(CLs, mses, label = f'Best {modeltype} model: L = {layer}, H = {emb}', color = colors[j])
        ax.set_xlabel('Context Length')
        ax.set_yscale('log')
        datatype = datatype[0].upper() + datatype[1:]
        ax.set_title(f'ICL on {datatype} Test Data')
        #make first letter of datatype capitalized
        ax.legend(loc = 'upper right')
    axs[0].set_ylabel('MSE')
    plt.savefig('figures/ICL_damped.png', bbox_inches = 'tight', dpi = 300)
    plt.show()
            

def get_model_hs_df(df, modeltype = 'underdamped', datatype = 'underdamped', traintest = 'train'):
    #gets hidden states for all models in modeldf, saves them, and returns a dataframe
    # with full data
    # saves hidden states for model with model. need to index layer and in layer position still. 
    modeldf = df[df['modeltype'] == modeltype]
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
        savepath = row['modelpath'][:-4]+f'_{datatype}_{traintest}_hss.pth'
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
    hsdf.to_csv(f'dfs/{modeltype}_{datatype}_{traintest}_model_hss.csv')
    return hsdf

def plot_attention(dfrow, datatype, traintest):
    model = load_model(dfrow, datatype)
    if 'linreg' in datatype:
        _, sequences = get_data(datatype, traintest)
        X, y = sequences[:, :-1], sequences[:, 1:]
        attns = model.return_attns(X)
        #attns = attns[:,:,:,0::2, 0::2]
    else:
        _, _, sequences, _, _ = get_data(datatype, traintest)
        X, y = sequences[:, :-1], sequences[:, 1:]
        attns = model.return_attns(X)
    #fig, axs = plt.subplots(1,attns.shape[0], figsize = (10,10))
    

    CL = attns.shape[3]
    CL = 32

    for i in range(attns.shape[0]):
        plt.figure(figsize=(10,10))
        attn = attns[i, :, 0, :CL, :CL]
        attn_mean = attn.mean(dim=0).detach().numpy()
        np.set_printoptions(precision=3)
        print(f'Layer {i+1}')
        print(attn_mean)
        # print attn_mean with 3 digits
    
        plt.imshow(attn_mean, cmap='viridis')
        # Annotate each square with its value
        for y in range(attn_mean.shape[0]):
            for x in range(attn_mean.shape[1]):
                if x>y:
                    continue
                mean = f'{attn_mean[y,x]:.2f}'
                print(attn_mean[y,x], mean)
                mean = mean[-2:]
                plt.text(x, y, f'{mean}', ha='center', va='center', color='white')
        # make xticks and yticks
        plt.xticks(range(CL), labels = [f'{i}' for i in range(CL)])
        plt.yticks(range(CL), labels = [f'{i}' for i in range(CL)])
        plt.ylabel('Neuron Number')
        plt.xlabel('Attending to Neuron Number')
        modelname = dfrow['modelpath'].split('/')[-1]
        plt.title(f'{modelname}\nattn-{i+1}')


        plt.show()
    
        
if __name__ == '__main__':
    
    #modeltype = 'underdamped'
    df = get_df_models()
    df = df[df['epoch'] == 20000]

    # for modeltype in ['underdamped', 'overdamped', 'damped']:
    #     for datatype in ['underdamped', 'overdamped', 'damped']:
    #         for traintest in ['train', 'test']:
    #             mdf = df[df['modeltype'] == modeltype]
    #             print(modeltype, datatype, traintest)
    #             plot_ICL(mdf, datatype = datatype, traintest = traintest, modeltype = modeltype)
                #get_model_hs_df(df, modeltype, datatype, traintest)
    # for modeltype in ['underdamped', 'overdamped', 'damped']:
    #     df = get_df_models()
    #     df = df[df['epoch'] == 20000]
    #     df = df[df['datatype'] == modeltype]
    #     df = df[df['emb'] < 64]
    #     for datatype in ['underdamped', 'overdamped']:
    #         plot_ICL(df, datatype = datatype, traintest = 'test', modeltype = modeltype)
    plot_ICL_damped()
    #plot_attention(df.iloc[0], 'undamped', 'train')
    #get_model_hs_df(df,'undamped', 'train')
    #plot_ICL(df, datatype = 'undamped', traintest = 'train')
    # df = df[df['emb'] == 16]
    # df = df[df['layer'] == 2]
    #plot_ICL(df, datatype = 'linreg1', traintest = 'test')
    #get_model_hs_df(df, datatype = 'linreg1cca', traintest = 'train')
    #get_model_hs_df(df)
    #plot_ICL(df, datatype = 'underdamped', traintest = 'train')