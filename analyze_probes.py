import pandas as pd
import torch
import time
from util import get_data, get_model_df, get_targetmethod_df, get_target_df, get_layer_pos

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
    #print(A[0])
    # ypred = (A @ X.unsqueeze(-1)).squeeze(-1)
    # se = (ypred - y)**2
    # mses = []
    # for i in range(len(CLs)):
    #     mses.append(A[:, i].mean())
    # print(mses)


if __name__ == '__main__':
    start = time.time()
    datatype = 'underdamped'
    traintest = 'train'
    df = readdf(datatype, traintest, 1, 16)
    predict_model_eAdt(df, datatype, traintest)