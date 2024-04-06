import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from util import load_model, get_data
import matplotlib.pyplot as plt
from inspector_models import LinearProbe, LinearDirect
from util import get_hidden_state, get_hidden_states, linear_multistep_coefficients
from sklearn.metrics import r2_score
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from tqdm import tqdm
import re
from sklearn.decomposition import PCA
from matplotlib.cm import get_cmap
from numerical_methods_theory import get_linear_multistep_pred
from math import factorial

def plot_positional_encodings(modeltype='underdamped', CL=65):
    # Assuming load_model is a function that loads the model
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    model.eval()
    posemb = model.positional_embeddings[:CL].detach()
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(posemb)
    
    # Create a colormap
    cmap = get_cmap('viridis', CL)
    
    # Scatter plot with labels and colormap
    for i in range(CL):
        plt.scatter(pcs[i, 0], pcs[i, 1], color=cmap(i))
        plt.text(pcs[i, 0], pcs[i, 1], str(i), fontsize=9)
    vars = pca.explained_variance_ratio_
    plt.title(f'Visualizations of Positional Embeddings {modeltype} Model \n {(vars[0]+vars[1])*100:.2f}% Var Captured')
    plt.xlabel(f'PC1 {vars[0]*100:.2f}%')
    plt.ylabel(f'PC2 {vars[1]*100:.2f}%')
    plt.show()

# DOES EVERYTHING WITH UNDERDAMPED DATA
def get_target_values(target, modeltype, dataset, traintest):
    # gets target values rk or lm
    # (# data points, CL)
    if dataset!='underdamped' or traintest!='train':
        print('BAD INPUTS')
        return
    terms = torch.load(f'data/{modeltype}spring_{dataset}_{traintest}_LM10.pth')
    terms.update(torch.load(f'data/{modeltype}spring_{dataset}_{traintest}_RK10.pth'))
    terms.update(torch.load(f'data/{modeltype}spring_{dataset}_{traintest}_mw.pth'))
    return terms[target]

def get_all_target_prec(prec, maxorder, modeltype, dataset, traintest):
    # gets all target values that start with prec
    terms = torch.load(f'data/{modeltype}spring_{dataset}_{traintest}_LM10.pth')
    terms.update(torch.load(f'data/{modeltype}spring_{dataset}_{traintest}_RK10.pth'))
    terms.update(torch.load(f'data/{modeltype}spring_{dataset}_{traintest}_mw.pth'))
    returndict = {}
    for t in terms:
        order = re.search(r'_[a-zA-Z](\d+)', t)
        if prec in t and (order is None or int(order.group(1)) < maxorder):
            returndict[t] = terms[t]
    return returndict




def getLinearProbe(hs, modeltype, targetname, layer, neuron, neuronall = False, run = True):
    # retrieves a stored linear probe and runs it on a hidden state
    if neuronall: # want the probe that was used on all neurons not a single state
        neuronstr = 'allneurons'
    else:
        neuronstr = f'neuron{neuron}'
    probe = LinearProbe(hs.shape[1])
    probepath = f'probes/{modeltype}/{targetname}_layer{layer}_{neuronstr}_Linear_probe.pth'
    probe.load_state_dict(torch.load(probepath))
    if run:
        target_pred = probe(hs).squeeze()
        return target_pred, probe
    else:
        return probe
   


def find_neuron_encoding(hs, targetvals, targetname, modeltype, layer, neuron, neuronall = False):
    # for a given hidden state, finds the R^2, MSE, and avg mag of the linear probe of the targetname
    target_pred, _ = getLinearProbe(hs, modeltype, targetname, layer, neuron, neuronall)
    r2 = r2_score(targetvals.detach().numpy(), target_pred.detach().numpy())
    criterion = nn.MSELoss()
    mse = criterion(target_pred, targetvals).detach()
    return r2, mse, target_pred.detach()

def plot_target_pred(targetname, modeltype = 'underdamped', dataset = 'underdamped', traintest = 'train', layer = 2, neuron = 7, colorby = 'gammas', plot = True):
    gammas, omegas, sequences, times, deltat =  get_data(datatype = dataset, traintest = traintest)
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    model.eval()
    hs = get_hidden_state(model, sequences, CL = 65, layer = layer, neuron = neuron)
    tvals = get_target_values(targetname, modeltype, dataset, traintest)[:, neuron]
    r2, mse, target_pred = find_neuron_encoding(hs, tvals, targetname, modeltype, layer, neuron)

    colorbys = {'gammas': gammas, 'omegas': omegas, 'deltat': deltat}
    if plot:
        plt.scatter(tvals, target_pred, label = f'Linear Probe R2 = {r2:.3f}, MSE = {mse:.2e}\nAvg Mag of {targetname} = {tvals.mean():.2e}', c = colorbys[colorby])
        plt.plot(tvals, tvals, label = 'Perfect Predictor', color = 'r')
        plt.title(f'{targetname} Linear Probe for Layer {layer} Neuron {neuron} Colored by {colorby}')
        plt.xlabel('Actual Target Vals')
        plt.ylabel('Predicted Target Vals (Probe)')
        plt.legend()
        plt.show()
    return r2, mse, tvals




def pca_neuron_target(targetname, modeltype = 'underdamped', dataset = 'underdamped', traintest = 'train', layer = 2, neuron = 7):
    '''
    PCAs the hidden states at a layer neuron number. Colors it by a certain target. 
    '''
    gammas, omegas, sequences, times, deltat =  get_data(datatype = dataset, traintest = traintest)
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    model.eval()
    hs = get_hidden_state(model, sequences, CL = 65, layer = layer, neuron = neuron)
    tvals = get_target_values(targetname, modeltype, dataset, traintest)[:, neuron]
    r2, mse, _ = find_neuron_encoding(hs, tvals, targetname, modeltype, layer, neuron, plot = True)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(hs)
    vars = pca.explained_variance_ratio_
    plt.scatter(pcs[:,0], pcs[:, 1], label = f'Linear Probe R2 = {r2:.3f}, MSE = {mse:.2e}\nAvg Mag of {targetname} = {tvals.mean():.2e}', c = tvals)
    plt.xlabel(f'PC1 {vars[0]*100:.2f}%')
    plt.ylabel(f'PC2 {vars[1]*100:.2f}%')
    plt.legend()
    plt.title(f'PCA of Layer {layer} Neuron {neuron} Hidden State\nColored w/ {targetname}')
    plt.show()

def evolve_r2mse_target(targetname, modeltype = 'underdamped', dataset = 'underdamped', traintest = 'train', layer = 2, CL = 65, plot = True, model = None, hss = None):
    gammas, omegas, sequences, times, deltat =  get_data(datatype = dataset, traintest = traintest)
    if model is None:
        model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    if hss is None:
        hss = get_hidden_states(model, sequences, CL = 65)
    tvals = get_target_values(targetname, modeltype, dataset, traintest)
    r2s, mses, neurons = [], [], []
    for neuron in range(CL):
        hs = hss[:, layer, neuron, :]
        tval = tvals[:, neuron]
        r2, mse, _ = find_neuron_encoding(hs, tval, targetname, modeltype, layer, neuron)
        r2s.append(r2)
        mses.append(mse)
        neurons.append(neuron)
    if plot:
        minr2s = [max(r2, 0) for r2 in r2s]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(neurons, minr2s,'o', color = 'b', label = 'R2s')
        ax1.set_ylabel('R2', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2.plot(neurons, mses,'o', color = 'r', label = 'MSEs')
        ax2.set_ylabel("MSE", color = 'r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_yscale('log')
        ax1.set_xlabel('Context Length')

        plt.title(f'Evolution of R2s and MSEs for {targetname} Linear Probe Layer {layer}')
        plt.show()
    return neurons, r2s, mses

def evolve_r2mse_alltargets(prec, maxorder, modeltype = 'underdamped', dataset = 'underdamped', traintest = 'train'):
    gammas, omegas, sequences, times, deltat =  get_data(datatype = dataset, traintest = traintest)
    terms = get_all_target_prec(prec, maxorder, modeltype = 'underdamped', dataset = 'underdamped', traintest = 'train')
    targetnames = list(terms.keys())
    nrows, ncols = len(terms.keys())//2, 2
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    hss = get_hidden_states(model, sequences, CL = 65)
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, sharey = True, sharex = True, figsize = (10,10))
    for i in range(nrows):
        for j in range(ncols):
            targetname = targetnames[i*ncols+j]
            print(targetname)
            neurons, r2s, mses = evolve_r2mse_target(targetname, modeltype = modeltype, dataset = dataset, traintest = traintest, layer = 2, plot = False, model = model, hss = hss)
            minr2s = [max(0, r2) for r2 in r2s]
            # Left y-axis for R2 scores
            ax1 = axs[i, j]
            ax1.scatter(neurons, minr2s, label=targetname, color='b')
            
            
            ax1.set_ylim(-0.05, 1.05)
            # Right y-axis for MSE scores
            ax2 = ax1.twinx()
            ax2.scatter(neurons, mses, color='r')
            ax2.set_yscale('log')

            if j == 0:
                ax1.set_ylabel('R2', color='b')
                ax1.tick_params(axis='y', labelcolor='b')
            else:
                ax2.set_ylabel('MSE', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
            ax1.set_title(targetname)
    #plt.title(prec)
    plt.show()


def pca_probe_target(targetname, modeltype = 'underdamped', CL = 65, plot = True):
    weights = []
    for neuron in range(CL):
        file = f'probes/{modeltype}/{targetname}_layer2_neuron{neuron}_Linear_probe.pth'
        probe = LinearProbe(16)
        probe.load_state_dict(torch.load(file))
        with torch.no_grad():
            weight = probe.l2.weight @ probe.l1.weight
            weight = weight.T.squeeze()
        weights.append(weight)
    weights = torch.stack(weights)
    if plot: 
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(weights)
        vars = pca.explained_variance_ratio_
        cmap = get_cmap('viridis', CL)
        for i in range(CL):
            plt.scatter(pcs[i, 0], pcs[i, 1], color=cmap(i))
            plt.text(pcs[i, 0], pcs[i, 1], str(i), fontsize=9)
        plt.xlabel(f'PC1 {vars[0]*100:.2f}%')
        plt.ylabel(f'PC2 {vars[1]*100:.2f}%')
        plt.title(f'PCA of {targetname} Probes Layer {2} \n {(vars[0]+vars[1])*100:.2f}% Var Captured')
        plt.show()
    return weights

def pca_probe_targets(prec, maxorder, takeoff = True, modeltype = 'underdamped', CL = 65, plot = True):
    terms = get_all_target_prec(prec, maxorder, modeltype = 'underdamped', dataset = 'underdamped', traintest = 'train')
    targetnames = list(terms.keys())
    allweights = None
    for targetname in targetnames:
        weights = pca_probe_target(targetname, modeltype = modeltype, CL = CL, plot = False)
        if allweights is None:
            allweights = weights
        else:
            allweights+=weights
    if takeoff:
        allweights = allweights[maxorder:]
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(allweights)
    vars = pca.explained_variance_ratio_
    cmap = get_cmap('viridis', allweights.shape[1])
    for i in range(allweights.shape[0]):
        plt.scatter(pcs[i, 0], pcs[i, 1], color=cmap(i))
        plt.text(pcs[i, 0], pcs[i, 1], str(i+maxorder), fontsize=9)
    plt.xlabel(f'PC1 {vars[0]*100:.2f}%')
    plt.ylabel(f'PC2 {vars[1]*100:.2f}%')
    plt.title(f'PCA of {prec} Up To Order {maxorder} Probes\n {(vars[0]+vars[1])*100:.2f}% Var Captured')
    plt.show()

def get_probe_predictions(prec, maxorder, modeltype = 'underdamped', dataset = 'underdamped', traintest = 'train', layer = 2):
    # gets all probe predictions and stores in data
    gammas, omegas, sequences, times, deltat =  get_data(datatype = dataset, traintest = traintest)
    X,y = sequences[:, :-1, :], sequences[:, 1:, :]
    print(X.shape, y.shape)
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    model.eval()
    targetnames = get_all_target_prec(prec, maxorder, modeltype = modeltype, dataset = dataset, traintest = traintest).keys()
    print(targetnames)
    hss = get_hidden_states(model, sequences, CL = 65)
    targetprobevals = {t:torch.zeros(X.shape[:2]) for t in targetnames}
    for neuron in range(X.shape[1]):
        for targetname in targetnames:
            hs = hss[:, layer, neuron, :]
            target_pred, _ = getLinearProbe(hs, modeltype, targetname, layer, neuron, neuronall = False)
            targetprobevals[targetname][:, neuron] = target_pred
    torch.save(targetprobevals, f'probepredictions/{modeltype}spring_{dataset}_{traintest}_{prec}{maxorder}probes.pth')

def lm_prediction(maxorder, modeltype = 'underdamped', dataset = 'underdamped', traintest = 'train', explicit = False):
    gammas, omegas, sequences, times, deltat =  get_data(datatype = dataset, traintest = traintest)
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    
    X,y = sequences[:, :-1, :], sequences[:, 1:, :]
    totest = {'actual_numerical_pred': 'data/underdampedspring_underdamped_train_LM10.pth', 'probe_numerical_pred': f'probepredictions/{modeltype}spring_{dataset}_{traintest}_LM10probes.pth'}
    result_preds = {test: {} for test in totest}
    #result_preds['model_pred'] = model(X)
    targets = ['x', 'v']
    for test in totest:
        y_backsteps = [torch.zeros(y.shape)] # implict step. not used for y
        f_backsteps = []
        probe_preds = torch.load(totest[test])
        f_backstep = torch.zeros((X.shape[0], X.shape[1], len(targets), len(targets)))
        for row in range(len(targets)): # implicit step
            for col in range(len(targets)):
                f_backstep[:,:,row,col] = probe_preds[f'LMf_{targets[row]}0{targets[col]}']
        f_backsteps.append(f_backstep.sum(dim=3)) # implict step (backstep = 0)

        for backstep in range(1,maxorder+1):
            y_backstep = torch.zeros(X.shape)
            f_backstep = torch.zeros((X.shape[0], X.shape[1], len(targets), len(targets)))
            for row in range(2):
                y_backstep[:,:,row] = probe_preds[f'LMy_{targets[row]}{backstep}']
                for col in range(2):
                    f_backstep[:,:,row,col] = probe_preds[f'LMf_{targets[row]}{backstep}{targets[col]}']
            y_backsteps.append(y_backstep)
            f_backsteps.append(f_backstep.sum(dim = 3))
        f_backsteps = torch.stack(f_backsteps)
        y_backsteps = torch.stack(y_backsteps)
        for order in range(1,maxorder+1):
            preds = get_linear_multistep_pred(order, f_backsteps, y_backsteps, explicit = True)
            result_preds[test][order] = preds
    if explicit:
        exp = 'explicit'
    else:
        exp = 'implicit'

    title = f'LM {exp} Predictions for {modeltype} Model on {dataset} {traintest} Data'
    plot_actualnumerical_vsprobe(model, X, y, title, result_preds, maxorder)

def rk_prediction(maxorder, modeltype = 'underdamped', dataset = 'underdamped', traintest = 'train'):
    gammas, omegas, sequences, times, deltat =  get_data(datatype = dataset, traintest = traintest)
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    X,y = sequences[:, :-1, :], sequences[:, 1:, :]
    totest = {'actual_numerical_pred': 'data/underdampedspring_underdamped_train_RK10.pth', 'probe_numerical_pred': f'probepredictions/{modeltype}spring_{dataset}_{traintest}_rk10probes.pth'}
    result_preds = {test: {} for test in totest}
    for test in totest:
        terms = torch.load(totest[test])
        coefs = []
        rk_terms = []
        
        for term in range(0, maxorder+1):
            rk_term = torch.zeros(y.shape)
            coefs.append(1/factorial(term))
            rk_term[:,:,0] = (terms[f'rk_x{term}x']+terms[f'rk_x{term}v'])*coefs[term]
            rk_term[:,:,1] = (terms[f'rk_v{term}x']+terms[f'rk_v{term}v'])*coefs[term]
            rk_terms.append(rk_term)
        rk_terms = torch.stack(rk_terms)
        for order in range(1,rk_terms.shape[0]):
            pred = rk_terms[:order+1].sum(dim=0)
            result_preds[test][order] = pred
    title = f'RK Predictions for {modeltype} Model on {dataset} {traintest} Data'
    plot_actualnumerical_vsprobe(model, X, y, title, result_preds, maxorder)

def mw_prediction(modeltype = 'underdamped', dataset = 'underdamped', traintest = 'train'):
    gammas, omegas, sequences, times, deltat =  get_data(datatype = dataset, traintest = traintest)
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    X,y = sequences[:, :-1, :], sequences[:, 1:, :]
    totest = {'actual_numerical_pred': 'data/underdampedspring_underdamped_train_mw.pth', 'probe_numerical_pred': f'probepredictions/{modeltype}spring_{dataset}_{traintest}_w2probes.pth'}
    result_preds = {test: {} for test in totest}
    for test in totest:
        matrixtarget = torch.load(totest[test])
        xpred = matrixtarget['w00'] + matrixtarget['w01']
        vpred = matrixtarget['w10'] + matrixtarget['w11']
        pred = torch.stack([xpred, vpred], dim = 2)
        result_preds[test][1] = pred
    title = f'MW Predictions for {modeltype} Model on {dataset} {traintest} Data'
    plot_actualnumerical_vsprobe(model, X, y, title, result_preds, 1)

def plot_actualnumerical_vsprobe(model, X, y, title, result_preds, maxorder):
    criterion = nn.MSELoss()
    fig, axs = plt.subplots(1,2, figsize = (10,5))
    modelpred = model(X)
    modelmses = [criterion(modelpred[:, :CL], y[:, :CL]).detach().numpy() for CL in range(1, X.shape[1])]
    for i, test in enumerate(result_preds.keys()):
        for order in range(1, maxorder+1):
            neurons, mses = [], []
            pred = result_preds[test][order]
            for neuron in range(order+1, X.shape[1]):
                mse = criterion(pred[:, order:neuron], y[:, order:neuron]).detach().numpy()
                neurons.append(neuron)
                mses.append(mse)
            axs[i].plot(neurons, mses, label = f'Order {order}')
        axs[i].plot(range(1,X.shape[1]), modelmses, color = 'k', lw = 2, label = 'Transformer Model')
        axs[i].set_title(test)
        axs[i].set_yscale('log')
        axs[i].set_xscale('log')
        axs[i].legend()
        axs[i].set_xlabel('Context Length')
        axs[i].set_ylabel('MSE')
    fig.suptitle(title)
    plt.show()

    # #model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    



if __name__ == '__main__':
    #plot_target_pred('LMf_x5v', colorby = 'omegas', neuron = 40)
    #evolve_r2mse_target('LMf_x5v')
    #pca_neuron_target('rk_x3x')
    #plot_positional_encodings()
    # precs = ['LMf_v', 'LMy_x', 'LMy_v', 'rk_x', 'rk_v']
    # for prec in precs:
    #     pca_probe_targets(prec, 5)
    #get_probe_predictions('LM', 10)
    lm_prediction(7, explicit = False)
    #evolve_r2mse_alltargets('w', 3)
    #get_probe_predictions('w', 2)
    #rk_prediction(5)
    #mw_prediction()