import torch
import matplotlib.pyplot as plt
from config import get_default_config
import numpy as np
import torch.nn as nn
from matplotlib.ticker import ScalarFormatter
from util import load_model, get_hidden_state, get_hidden_state_old, set_seed, get_data
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from inspector_models import NonLinearProbe, probe_hiddenstate, probe_hiddenstates, LinearProbe
from scipy.stats import linregress
from numerical_methods_theory import rk, general_linear_multistep



config = get_default_config()


def plot_loss_curves(file = 'models/spring_16emb_2layer_10CL_20000epochs_0.001lr_64batch_losses.pth'):
    losses = torch.load(file)
    train_losses = losses['train_losses']
    test_in_losses = losses['test_in_losses']
    test_out_losses = losses['test_out_losses']

    epochs = torch.tensor(range(len(train_losses)))*10

    plt.plot(epochs, train_losses, c = 'r', label = 'Train Loss')
    plt.plot(epochs, test_in_losses, c = 'b', label = 'Test-In Loss')
    plt.plot(epochs, test_out_losses,'b--', label = 'Test-Out Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.title(f'Loss Curves\n{file}')
    plt.legend()
    plt.show()



def loss_deltat_omega_relationship(model, CL = 10):
    """
    We want to see how the loss varies with deltat and omega,
    the two effective parameters we give our model.
    Note that x = cos(wt), v = -wsin(wt). where t = deltat * i for integer i
    """

    datadict = torch.load('data/spring_data.pth')
    traindata = datadict['sequences_train'][:, :CL+1, :]
    div = int(0.8*traindata.shape[0])
    traindata = traindata[:div]
    trainomegas = datadict['train_omegas'][:div]
    traintimes = datadict['train_times'][:div, :CL+1]
    train_deltat = traintimes[:,1] - traintimes[:,0]
    Xtrain,ytrain = traindata[:,:-1,:],traindata[:,1:,:]

    testdata = datadict['sequences_test'][:, :CL+1, :]
    testomegas = datadict['test_omegas']
    testtimes = datadict['test_times'][:, :CL+1]
    test_deltat = testtimes[:,1] - testtimes[:,0]
    Xtest,ytest = testdata[:,:-1,:],testdata[:,1:,:]

    with torch.no_grad():
        ytrain_pred = model(Xtrain)
        MSE_train = (ytrain_pred - ytrain)**2
        MSE_train = torch.mean(MSE_train, dim=(1, 2)).numpy()  # Squeeze and convert to numpy

        ytest_pred = model(Xtest)
        MSE_test = (ytest_pred - ytest)**2
        MSE_test = torch.mean(MSE_test, dim=(1, 2)).numpy()  # Squeeze and convert to numpy

        # Assuming trainomegas and testomegas are numpy arrays and train_deltat and test_deltat are the corresponding delta T values
        scatter = plt.scatter(trainomegas, train_deltat, c=MSE_train, cmap='viridis')
        plt.scatter(testomegas, test_deltat, c=MSE_test, cmap='viridis')
        plt.axvline(x = torch.min(trainomegas).item(), label = 'Min Training Omega',color = 'r', linestyle = '--')
        plt.axvline(x = torch.max(trainomegas).item(), label = 'Max Training Omega', color = 'r', linestyle = '--')

        plt.xlabel('Omega')
        plt.ylabel('Delta T')
        plt.title('MSE vs DeltaT and Omega')
        plt.legend()

        cbar = plt.colorbar(scatter)
        cbar.set_label('Mean Squared Error (MSE)')

        # Use scientific notation for color bar tick labels
        cbar.formatter.set_scientific(True)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

        plt.show()

def test_ICL(models, evaluate):
    """
    This function `test_ICL` evaluates a model's performance using mean squared error on different
    context lengths for in-distribution and out-of-distribution test sets.
    models is the list of models we want to test
    evaluat is a dict of data_name: data 
    """

    criterion = nn.MSELoss()
    #print(config.max_seq_length)
    fig, axs = plt.subplots(1, len(evaluate.keys()), figsize = (15,7))
    colors = ['b', 'r', 'g', 'k']
    for j, model_key in enumerate(models.keys()):
        model = models[model_key]
        for i,evaluate_key in enumerate(evaluate.keys()):
            MSEs = []
            CLs = []
            for CL in range(1,65):
                print(CL)
                evaluate_data = evaluate[evaluate_key][:, :CL+1, :]
                X, y = evaluate_data[:100,:-1,:],evaluate_data[:100,1:,:]
                with torch.no_grad():
                    y_pred = model(X)
                    MSE = criterion(y_pred, y)
                CLs.append(CL)
                MSEs.append(MSE.item())
            xlog = np.log(CLs)
            ylog = np.log(MSEs)
            slope, intercept, r_value, p_value, std_err = linregress(xlog, ylog)
            MSEpred = np.exp(slope*xlog+intercept)
            axs[i].plot(CLs, MSEs, label = f'{model_key} model\nlog(MSE) = {slope:.2f}log(CL)+{intercept:.2f}, R^2 = {r_value**2:.2f}', color = colors[j])
            axs[i].plot(CLs, MSEpred, color = colors[j], linestyle = '--')
    for i, ax in enumerate(axs):
        ax.set_title(f'Test on {list(evaluate.keys())[i]} Data')
        ax.set_xlabel('Context Length')
        ax.set_ylabel('MSE')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()
    plt.show()

            
    # # find linear relationship between log of CLs and log of MSEs
    # slope, intercept, r_value, p_value, std_err = linregress(xlog, ylog)
    # ypredlog = slope*xlog+intercept
    # ypred = np.exp(ypredlog)


    # axs[0].plot(CLs, MSEs_train, c = 'b', label = f'Test Set - In Distribution')
    # axs[0].plot(CLs, ypred, c = 'b',linestyle = '--', label = f'log(MSE) = {slope:.2f}log(CL)+{intercept:.2f}, R^2 = {r_value**2:.2f}')
    # axs[1].plot(CLs, MSEs_test, c = 'r', label = 'Test Set - Out of Distribution')
    # for i in range(2):
    #     axs[i].set_xlabel('Context Length')
    #     axs[i].set_ylabel('MSE')
    #     axs[i].legend()
    #     axs[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    #     axs[i].set_yscale('log')
    #     axs[i].set_xscale('log')
    # fig.suptitle('MSE vs CL In Distribution Test\nTrained on 65CL')
    # plt.show()


def analyze_hiddenstates(model, target = 'omegas'):
    CL = 10
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

    # Calculate correlation coefficients
    correlations = []
    for layer in range(hidden_states.shape[1]):  # Iterate over layers
        layer_correlations = []
        for state in range(hidden_states.shape[2]):  # Iterate over hidden states
            state_correlations = []
            for dim in range(hidden_states.shape[3]):  # Iterate over dimensions
                feature = hidden_states[:, layer, state, dim].unsqueeze(1).numpy()  # Convert to numpy array
                correlation = mutual_info_regression(feature, target_vals.numpy())[0]  # CALCULATE MI
                #correlation, _ = pearsonr(feature, target_vals.numpy())  # Calculate Pearson correlation coefficient
                state_correlations.append(abs(correlation))  # Store absolute value of correlation
                
            layer_correlations.append(np.mean(state_correlations))  # Sum of absolute correlations for each state
        correlations.append(layer_correlations)
    correlations = np.array(correlations)

    layer_corr = np.mean(correlations, axis=1)  # Average correlation for each layer
    hs_corr = np.mean(correlations, axis=0)  # Average correlation for each hidden state


    # Plotting
    fig, ax = plt.subplots(figsize=(12, 5))
    cax = ax.matshow(correlations, cmap='viridis')
    cbar = fig.colorbar(cax)
    cbar.set_label('Avg. Mag of Correlation Across Embedding')  # Set label for color bar

    # Set x and y ticks manually
    ax.set_xticks(range(correlations.shape[1]))  # Assuming you have 10 hidden states
    ax.set_yticks(range(correlations.shape[0]))   # Assuming you have 3 layers
    ax.set_xticklabels([f'{i}\n{hs_corr[i]:.2f}' for i in range(correlations.shape[1])])
    ax.set_yticklabels([f'After Pos Emb\n{layer_corr[0]:.2f}']+[f'After Layer {i}\n{layer_corr[i]:.2f}' for i in range(1,correlations.shape[0])])
    ax.set_xlabel('Hidden State')
    #ax.set_ylabel('Layer')

    # Add text annotations with the correlation values
    for i in range(correlations.shape[0]):
        for j in range(correlations.shape[1]):
            ax.text(j, i, f'{correlations[i, j]:.2f}', ha='center', va='center', color='white')

    ax.set_title(f'Pearson Correlation of Hidden States with {target}\nTest on CL = {CL}, Train w/ {hidden_states.shape[1] - 1}Layers, {hidden_states.shape[-1]}Emb, 65CL')
    plt.show()
    return np.array(correlations)  # Shape: [3, 10], representing overall correlation for each hidden state in each layer

def plot_neuron_old(model, layer = 0, neuron = 0, target = 'omegas', CL = 10):

    hidden_states, target_vals = get_hidden_state_old(model, CL, layer, neuron, target)
    corrs = []
    for i in range(hidden_states.shape[1]):
        corr_interim, _ = pearsonr(hidden_states[:,i].numpy(), target_vals.numpy())
        corrs.append(corr_interim)
    corrs = np.abs(np.array(corrs))
    # Calculate mutual information

    corr = np.mean(corrs)
    
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(hidden_states)
    fig, ax = plt.subplots(figsize = (10,6))
    cax = ax.scatter(pcs[:, 0], pcs[:, 1], c=target_vals, cmap='viridis')
    cbar = fig.colorbar(cax)
    cbar.set_label(f'{target}\nAvg Corr on Emb = {corr:.2f}')  # Set label for color bar

    vars = pca.explained_variance_ratio_
    # set label on colorbar
    ax.set_xlabel(f'PC1 {vars[0]*100:.2f}% Var')
    ax.set_ylabel(f'PC2 {vars[1]*100:.2f}% Var')
    ax.set_title(f'PCA of Hidden States in Layer {layer}, Neuron {neuron} w/ {target}\nTest on CL = {CL}, Train w/ {hidden_states.shape[1] - 1}Layers, {hidden_states.shape[-1]}Emb, 65CL\n{100*np.sum(vars):.1f}% Var Captured')
    plt.show()

def plot_neuron(model, modelname, data, targetdict, layer = 0, neuron = 0, CL = 10):
    # dict of name:target we want to plot neuron against

    hidden_states = get_hidden_state(model, data, CL, layer, neuron)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(hidden_states)


    for target_name in targetdict.keys():


        target_vals = targetdict[target_name]
        r2 = probe_hiddenstate(model, data,linear =True, target_name = target_name, target_vals = target_vals, layer = layer, neuron = neuron, epochs = 20000)
        
        
        fig, ax = plt.subplots(figsize = (10,6))
        cax = ax.scatter(pcs[:, 0], pcs[:, 1], c=target_vals, cmap='viridis')
        cbar = fig.colorbar(cax)
        cbar.set_label(f'{target_name}\nR^2 of Non Linear Probe= {r2:.3f}')  # Set label for color bar

        vars = pca.explained_variance_ratio_
        # set label on colorbar
        ax.set_xlabel(f'PC1 {vars[0]*100:.2f}% Var')
        ax.set_ylabel(f'PC2 {vars[1]*100:.2f}% Var')
        ax.set_title(f'{modelname}\nPCA of Hidden States in Layer {layer}, Neuron {neuron} w/ {target_name}\nTest on CL = {CL}, Train w/ {hidden_states.shape[1] - 1}Layers, {hidden_states.shape[-1]}Emb, 65CL\n{100*np.sum(vars):.1f}% Var Captured')
        plt.show()


def generate_targets(omega, gamma, deltat):
    targetdict = {}
    for o_exp in range(5):
        for g_exp in range(5):
            for d_exp in range(5):
                target = (omega**o_exp)*(gamma**g_exp)*(deltat**d_exp)
                target_name = f'omega0^{o_exp}gamma^{g_exp}deltat^{d_exp}'
                targetdict[target_name] = target
    return targetdict


def create_probes(models, targetdicts, datatype = 'underdamped', traintest = 'train'):
    for modelkey in models.keys():
        gammas, omegas, sequences, times, deltat = get_data(datatype = datatype, traintest = traintest)

        for targetdict in targetdicts:
            # save target_dict

            #generate_targets(omega0, gamma, deltat)
            model = models[modelkey]
            print(modelkey)
            #targets = []
            avg_magsv = []
            r2sv = []
            avg_magsx = []
            r2sx = []
            layer, neuron = 2, 0

            data = torch.load('data/dampedspring_data.pth')
            data = torch.cat((data[f'sequences_test_{datatype}'], data[f'sequences_train_{datatype}']))
            print(data.shape)
            print(sequences.shape)
            for target_name in targetdict.keys():
                target_vals = targetdict[target_name]#[:, neuron]
                probe_hiddenstates(model, modelkey, sequences, target_vals, target_name, linear = True, CL = 65, epochs = 20000, num_layers = 2)


def deltat_loss_relationship(model):
    key = 'underdamped'
    data = torch.load('data/dampedspring_data.pth')
    sequences = data[f'sequences_train_{key}']
    deltat = data[f'times_train_{key}'][:, 1] - data[f'times_train_{key}'][:, 0]
    indices = torch.randperm(sequences.size(0))
    sequences = sequences[indices][:500]
    deltat = deltat[indices][:500]
    X,y = sequences[:,:-1,:],sequences[:,1:,:]

    with torch.no_grad():
        y_pred = model(X)
        MSE = (y_pred - y)**2
        dims = torch.arange(1, MSE.shape[1]+1)
        dims = dims.repeat(MSE.shape[0], 1).flatten()
        print(deltat.shape)
        #deltat = deltat.repeat(MSE.shape[1])
        deltat = deltat.repeat(MSE.shape[1], 1).numpy()
        deltat = deltat.flatten('F')

        # repeat deltat for MSE.shape[1]
        # repeat first digit of deltat n times and do for all digits
        #print(dims.squeeze())
        MSE = torch.flatten(torch.mean(MSE, dim = 2)).numpy()
        

    # color bar by dims
    plt.scatter(deltat, MSE) #c = deltat, cmap = 'viridis')
    # show colorbar
    cbar = plt.colorbar()
    cbar.set_label('Delta T')
    #plt.scatter(deltat, MSE, c = 'b')
    plt.xlabel('Delta T')
    plt.ylabel('MSE')
    plt.title(f'MSE vs Delta T for {key} on Train Set')
    plt.yscale('log')
    plt.show()



if __name__ == '__main__':
    seed = 0
    set_seed(seed)
    underloss = 'models/springunderdamped_16emb_2layer_65CL_20000epochs_0.001lr_64batch_losses.pth'
    undermodel = 'models/springunderdamped_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth'
    underdamped = load_model(file = undermodel)

    overloss = 'models/springoverdamped_16emb_2layer_65CL_20000epochs_0.001lr_64batch_losses.pth'
    overmodel = 'models/springoverdamped_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth'
    overdamped = load_model(file = overmodel)

    dampedloss = 'models/springdamped_16emb_2layer_65CL_20000epochs_0.001lr_64batch_losses.pth'
    dampedmodel = 'models/springdamped_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth'
    damped = load_model(file = dampedmodel)
    
    config.n_layer = 3
    under3loss = 'models/springunderdamped_16emb_3layer_65CL_20000epochs_0.001lr_64batch_losses.pth'
    under3model = 'models/springunderdamped_16emb_3layer_65CL_20000epochs_0.001lr_64batch_model.pth'
    underdamped3 = load_model(file = under3model, config = config)
    #plot_loss_curves(file = under3loss)
    losspaths = [underloss, overloss, dampedloss, under3loss]
    # for losspath in losspaths:
    #     plot_loss_curves(file = losspath)
    data = torch.load('data/dampedspring_data.pth')
    evaluate = {'underdamped': data['sequences_test_underdamped'], 
                'overdamped': data['sequences_test_overdamped'],
                }
    models = {'underdamped': underdamped,
              #'3-layer underdamped': underdamped3, 
            #   'overdamped':overdamped, 
            #   'damped':damped
              }
    #deltat_loss_relationship(underdamped)
    datatype, traintest = 'underdamped', 'train'
    order = 10
    rk(order, datatype = datatype, traintest = traintest, plot = False)
    general_linear_multistep(order, datatype = datatype, traintest = traintest, plot = False)
    LM = torch.load(f'data/underdampedspring_{datatype}_{traintest}_LM10.pth')
    RK = torch.load(f'data/underdampedspring_{datatype}_{traintest}_RK10.pth')
    targetdicts = [LM, RK]
    create_probes(models, targetdicts)
    
    #test_ICL(models, evaluate)
    #create_probes(models)

    
            # r2 = probe_hiddenstate(model = model, modelname = modelkey, data = data, target_vals = target_vals, target_name = target_name, linear = True, CL = 10, epochs = 10000, layer = layer, neuron = neuron, plot = False)
            # avg_mag = target_vals.abs().mean().item()
            # print(f'{target_name}: R^2 = {r2:.3f}, Avg Mag = {avg_mag:.3e}')
            # if target_name[0] == 'v':
            #     avg_magsv.append(avg_mag)
            #     r2sv.append(r2)
            # elif target_name[0] == 'x':
            #     avg_magsx.append(avg_mag)
            #     r2sx.append(r2)
            #targets.append(target_name)
        # plt.scatter(avg_magsx, r2sx, color = 'b', label = 'x(t+dt) Taylor Expansion')
        # plt.scatter(avg_magsv, r2sv, color = 'r', label = 'v(t+dt) Taylor Expansion')
        # plt.title(f'{modelkey} Model\nR^2 of Linear Probe vs Avg Magnitude of Taylor Expansion Term\nlayer = {layer}, neuron = {neuron}')
        # plt.xscale('log')
        # plt.xlabel('Avg Magnitude of Taylor Expansion Term')
        # plt.ylabel('R^2 of Linear Probe')
        # plt.legend()
        # plt.show()

        #plot_neuron(models[modelkey], modelkey+' Model', data, targetdict, layer = 2, neuron = 9, CL = 10)
        
    #model = load_model(file = modelpath)
    #plot_neuron(model)
    #test_ICL(models, evaluate)
    #plot_loss_curves(file = losspath)


    #test_ICL(model)
    # for layer in [0]:
    #     for neuron in range(1):
    #         plot_neuron(model,layer = layer, neuron = neuron, target = 'omegas')
    #pca.explained_variance_ratio_

    #loss_deltat_omega_relationship(model)
    #plot_loss_curves(file = losspath)