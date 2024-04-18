import pandas as pd
import torch
import time
from util import get_data, get_model_df, get_targetmethod_df, get_target_df, get_layer_pos, load_model_file, get_log_log_linear
from config import linreg_config
import matplotlib.pyplot as plt
import numpy as np

def readdf(datatype, traintest, modellayers, modelemb):
    return pd.read_csv(f'dfs/{datatype}_{traintest}_m{modellayers}e{modelemb}probes.csv', index_col=0)

def predict_model_eAdt(df, datatype, traintest):
    gammas, omegas, sequences, times, deltat = get_data(datatype, traintest)
    hss = torch.load(df['h-hspath'].values[0])
    X,y = sequences[:, :-1], sequences[:, 1:]
    x,v = X[:, :, 0], X[:, :, 1]
    df = df.sort_values('p-CL')
    CLs = df['p-CL'].unique()
    df_ea = get_targetmethod_df(df, 'eA')
    A = torch.zeros((X.shape[0],len(CLs), 2, 2))
    AX = torch.zeros((len(CLs), 2, 2))
    df = df.sort_values('p-mse')
    for CL in CLs:
        minidf = df[df['p-CL'] == CL]
        for row in range(2):
            for col in range(2):
                target = f'eAdt{row}{col}'
                minidft = get_target_df(minidf, target)
                print(f'A{row}{col}', minidft['p-mse'].mean(), 'mean')
                minidft = minidft.iloc[0]
                
                layerpos, inlayerpos = minidft['h-layerpos'], minidft['h-inlayerpos']
                hs = hss[layerpos][inlayerpos][:, CL].detach().numpy()
                bestprobe = minidft['p-savepath']
                clf = torch.load(bestprobe)
                value = clf.predict(hs)
                A[:, CL, row, col] = torch.tensor(value)

def intervene(mlayer, membd, hlayerpos, hinlayerpos):
    #mlayer and membd define the model to intervene on
    #hlayerpos and hinlayerpos define the layer to intervene on
    # tries to make all weights 0.5
    weights, sequences = get_data(datatype = 'linreg1', traintest = 'train')
    X,y = sequences[:, :-1], sequences[:, 1:]
    df = pd.read_csv('dfs/proberesults/reverseproberesults_rlinreg1_train/proberesults_rlinreg1_train_0.csv', index_col=0)
    # only df where m-layer = 3
    df = df[df['m-layer'] == mlayer]
    df = df[df['m-emb'] == membd]
    df = df[df['h-inlayerpos'] == hinlayerpos]
    df = df[df['h-layerpos'] == hlayerpos]
    # sort df by p-CL
    df = df.sort_values('p-CL')
    # iterate through df

    targetname = df['p-targetname'].unique()
    print(targetname)
    assert len(targetname) == 1
    maxexp = int(targetname[0][-1])
    newweight = torch.zeros(weights.shape[0], maxexp)
    
    for i in range(1,maxexp+1):
        newweight[:,i-1] = 0.5**i

    insert = {}

    y05 = X[:,0::2]*0.5
    # intersperse X with y05
    y05 = torch.cat((X[:,0::2], y05), dim=2)
    y05 = y05.view(X.shape[0], -1)[:,1:]

    for index, row in df.iterrows():
        path = row['p-savepath']
        print(path)
        CL = row['p-CL']
        probe = torch.load(path)
        newhs = probe.predict(newweight)
        print(newhs)
        insert[CL] = newhs

    insert = {hlayerpos:{hinlayerpos: insert}}
    modelpath = df['m-modelpath'].unique()[0]
    config = linreg_config()
    config.n_layer = mlayer
    config.n_embd = membd
    config.max_seq_length = 131
    model = load_model_file(modelpath,config)
    model.eval()
    y = model.forward(X, insertall = insert)
    mse = (y[:,0::2].squeeze() - y05[:,0::2])**2
    mse = mse.detach().numpy()
    CLs = np.array(list(range(2, y.shape[1]+1)))
    mse = [mse[:,1:i].mean() for i in CLs]
    #CLs, mse = CLs, mse
    # slope, intercept, r_value = get_log_log_linear(CLs, mse)
    # plt.plot(CLs, mse, label = f'log(MSE) = {slope:.2f}log(CL) + {intercept:.2f}, R^2 = {r_value**2:.2f}', color = 'b')
    # plt.xlabel('CL')
    # plt.ylabel('MSE')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.legend()
    
    # plt.title('MSE vs CL for Intervened Model')
    # plt.show()


    whats = (y[:,0::2]/X[:,0::2])[:,1:].detach().abs()
    whats = whats
    print(whats.flatten())
    xmin,xmax = 0,1
    #xmin, xmax = xmin**maxexp, xmax**maxexp
    plt.hist(whats.flatten(), density = True, bins = 20, range = (xmin, xmax))
    # find how many whats fall between 0.4 and 0.6
    inrangepercent = len(whats[(whats > xmin) & (whats < xmax)].flatten()) / len(whats.flatten())
    # set xmin
    
    plt.title(f'Intervening on {hinlayerpos}-{hlayerpos} to Change w = 0.5 for all Time Series\n{inrangepercent*100:.1f}% of w in range {xmin} to {xmax}')
    plt.xlabel(rf'Observed $|w|$ from intervened model')
    plt.ylabel('Frequency')
    plt.xlim(xmin, xmax)

    plt.show()

    #plot points with error barsC
    # CLs = list(range(1,whats.shape[1]+1))
    # wmeans = whats.mean(dim = 0).flatten()
    # werrors = whats.std(dim = 0).flatten()
    # print(wmeans.shape, werrors.shape, len(CLs))
    # plt.errorbar(CLs, wmeans, yerr = werrors, fmt = 'o')
    # plt.axhline(0.5,color = 'r', linestyle = '--')
    # plt.ylim(0.4, 0.6)
    # plt.title('Intervening on attn-3 to Change w = 0.5 for all Time Series\nMean and Standard Deviation of w for each CL')
    # plt.xlabel('CL')
    # plt.ylabel('w')
    # plt.show()


def ols_best_loss():
    CL = 65
    arith_sum = 0
    mses = []
    CLs = []
    for i in range(1,CL+1):
        arith_sum+=0.001/i
        mses.append(arith_sum/i)
        CLs.append(i)
    slope, intercept, r_value = get_log_log_linear(CLs, mses)
    plt.plot(CLs, mses, label = f'log(MSE) = {slope:.2f}log(CL) + {intercept:.2f}, R^2 = {r_value**2:.2f}')
    plt.xlabel('CL')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('OLS Best MSE Loss vs CL')
    plt.legend()
    plt.show()




if __name__ == '__main__':
    intervene(3, 2, 1, 'attn')
