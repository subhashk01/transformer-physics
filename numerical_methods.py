import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from util import load_model
import matplotlib.pyplot as plt
from inspector_models import getLinearProbe, LinearProbe, LinearDirect
from util import get_hidden_state, get_hidden_states, euler_to_term, matrix_weights_to_term
from sklearn.metrics import r2_score
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from tqdm import tqdm

def find_neuron_encoding(hidden_states, targetvals, targetname, modeltype, layer, neuron, neuronall = False):
    # for a given hidden state, finds the R^2, MSE, and avg mag of the linear probe of the targetname
    hs = hidden_states[:, layer, neuron, :]
    target_pred = getLinearProbe(hs, modeltype, targetname, layer, neuron, neuronall)
    r2 = r2_score(targetvals.detach().numpy(), target_pred.detach().numpy())
    criterion = nn.MSELoss()
    mse = criterion(target_pred, targetvals).detach()
    return r2, mse

def plot_R2_firsths_euler(modeltype = 'underdamped', dataset = 'underdamped', layer = 2, neuron = 64):
    # for any neuron, plots how well the different taylor expansion terms are encoded in that hidden state
    # plots encoding (as a function of average magnitude of term) vs R^2 of linear probe. makes it comparable vs other terms

    plt.rcParams['text.usetex'] = True
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    model.eval()
    target_dict = torch.load(f'data/euler_terms_{dataset}.pth')
    sequences = target_dict['sequences']
    colors = 'brgcmyk'
    plt.figure(figsize = (10,5))

    hidden_states = get_hidden_states(model, sequences, CL = neuron+1)

    for e, euler_target in enumerate(['x','v']):
        avg_mags = []
        R2s = []
        targetnames = [f'{euler_target}{i}{term}' for i in range(5) for term in ['x','v']]
        for targetname in targetnames:
            if targetname in target_dict.keys():
                target = target_dict[targetname][:,neuron]
            else:
                continue
            r2, mse = find_neuron_encoding(hidden_states, target, targetname, modeltype, layer, neuron)
            R2s.append(r2)
            avg_mags.append()
            targetnames.append(targetname)
        #threshold all R2s values below 0 to 0
        R2s_adj = [max(0,r2) for r2 in R2s]
        
        plt.scatter(avg_mags, R2s_adj, color = colors[e], label = rf'${euler_target}(t+dt)$ Taylor Expansion Terms\\$R^2$ sum = {sum(R2s_adj):.2f}')
        # add text to scatter points with targetnames
        for i, txt in enumerate(targetnames):
            plt.annotate(txt, (avg_mags[i]*1.1, R2s_adj[i]))
    plt.xlabel('Average Magnitude of Term')
    plt.xscale('log')
    plt.ylabel(r'$R^2$ of Linear Probe of Term')
    plt.legend()
    plt.title(rf'$R^2$ of Linear Probes for Taylor Expansion\\{modeltype} Model, Layer {layer}, Neuron {neuron}')
    plt.show()


def plot_R2evolve_euler(testtype = 'euler',modeltype = 'underdamped', dataset = 'underdamped', CL = 65, layer = 2, R2 = True, neuronall = False):
    # plots how the R2 score of the linear probe evolves as the context length increases
    # uses probes stored at keys of mappings, which is a list of mapping, which is a dict that has probename: latex formula 
    # testtype can be either 'euler' or 'matrix_weights'
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    model.eval()
    plt.rcParams['text.usetex'] = True
    if testtype == 'euler': # want to see how well taylor expansion is encoded
        target_dict = torch.load(f'data/euler_terms_{dataset}.pth') 
        mapping = euler_to_term()
    else: # want to see how well max's matrix weights are encoded
        target_dict = torch.load(f'data/matrix_terms_{dataset}.pth')
        mapping = matrix_weights_to_term()

    colors = 'br'
    plt.figure(figsize = (10,5))
    # want to see encoding only in last layer bc this is ultimately what's used to do computation

    str = '$R^2$' if R2 else 'MSE'
    sequences = target_dict['sequences']

    sequences = sequences[:,:CL+1,:]
    hidden_states = get_hidden_states(model, sequences, CL = CL)

    for e, euler_target in enumerate(['x','v']):

        if testtype == 'euler':
            num_rows = 5
            targetnames = [f'{euler_target}{i}{term}' for i in range(num_rows) for term in ['x','v']]
        else:
            num_rows = 1
            if euler_target == 'x': 
                targetnames = ['w00', 'w01']
            else: 
                targetnames = ['w10', 'w11']
        fig, axs = plt.subplots(num_rows,2, sharey = R2, sharex = R2, figsize = (10,10))
        term_num = 0
        for targetname in targetnames:
            
            col_num = term_num % 2
            row_num = term_num // 2
            if testtype == 'euler':
                ax = axs[row_num, col_num]
            else: # take special case bc num_rows = 1
                ax = axs[col_num]
                ax.set_xlabel('Context Length')
            if col_num==0:
                ax.set_ylabel(rf'{str}')
            if row_num == axs.shape[0]-1:
                ax.set_xlabel('Context Length')
            neurons = list(range(CL))
            R2s = []
            avg_mags = []
            MSEs = []
            for neuron in neurons:
                if targetname in target_dict.keys():
                    target = target_dict[targetname][:,neuron]
                else:
                    continue
                r2, mse = find_neuron_encoding(hidden_states, target, targetname, modeltype, layer, neuron, neuronall = neuronall)
                R2s.append(r2)
                avg_mags.append(target.abs().mean())
                MSEs.append(mse)
                print(f'{targetname}: Neuron {neuron}, MSE = {mse:.2e}, R^2 = {r2:.2f}')
            
            if len(R2s):
                print(targetname, row_num, col_num)
                R2s = [max(0,r2) for r2 in R2s]
                metric = R2s if R2 else MSEs
                ax.plot(neurons, metric, color = colors[e], marker = 'o', label = rf'${targetname}: {mapping[targetname]}$ \\ Avg Mag = {target.abs().mean().item():.2e}\\Avg MSE = {np.mean(MSEs):.2e}')
                print(neurons)
                ax.legend()

            term_num+=1
            if R2:
                ax.set_ylim(-0.2,1.2)
            else:
                ax.set_yscale('log')
                if not neuronall:
                    ax.set_xscale('log')
                    
        fig.suptitle(rf'Evolution of {str} of Linear Probes for {testtype} {euler_target} Terms \\ {modeltype} Model, Layer {layer}')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        
        
def probe_allneurons_target(testtype = 'euler', modeltype = 'underdamped', dataset = 'underdamped', CL = 65, layer = 2, epochs = 10000):
    # basically, we want to find a single linear layer that best approximates the target for all neurons
    # works on either euler or matrix weights
    if testtype == 'euler': # want to see how well taylor expansion is encoded
        target_dict = torch.load(f'data/euler_terms_{dataset}.pth') 
        mapping = euler_to_term()
    else: # want to see how well max's matrix weights are encoded
        target_dict = torch.load(f'data/matrix_terms_{dataset}.pth')
        mapping = matrix_weights_to_term()
    targetnames = mapping.keys()
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    sequences = target_dict['sequences']
    hidden_states = get_hidden_states(model, sequences, CL = CL)[:,layer, :, :]
    print(hidden_states.shape)
    hshape0, hshape1 = hidden_states.shape[0], hidden_states.shape[1]
    hidden_states = hidden_states.view(hshape0*hshape1, hidden_states.shape[2])
    for targetname in targetnames:
        probe = LinearProbe(hidden_states.shape[1], 1)   
        optimizer = optim.Adam(probe.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        target = target_dict[targetname][:,:hshape1]
        target = target.reshape(target.shape[0] * target.shape[1])
        epoch_pbar = tqdm(range(epochs), desc='Training Probe')
        indices = torch.randperm(hidden_states.shape[0])
        use = 10000
        
        hs, target = hidden_states[indices][:use], target[indices][:use]
        div = int(hs.shape[0] * 0.8)

        hs_train, target_train = hs[:div], target[:div]
        hs_test, target_test = hs[div:], target[div:]
        print(targetname)
        for epoch in epoch_pbar:
            optimizer.zero_grad()
            output = probe(hs_train).squeeze()
            loss = criterion(output, target_train)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                test_loss = criterion(probe(hs_test).squeeze(), target_test)
            epoch_pbar.set_description(f'Epoch {epoch}')
            epoch_pbar.set_postfix({'Train Loss': f'{loss.item():.2e}',
                        'Test Loss': f'{test_loss.item():.2e}'})
        torch.save(probe.state_dict(), f'probes/{modeltype}/{targetname}_layer{layer}_allneurons_Linear_probe.pth')


def check_out_modelattention(modeltype = 'underdamped'):
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    model.eval()

    data = torch.load('data/dampedspring_data.pth')
    data = torch.cat((data[f'sequences_test_damped'], data[f'sequences_train_damped']))
    attns = model.return_attns(data)
    return attns

def plot_attention_maps(modeltype='underdamped', CL=10):
    attns = check_out_modelattention(modeltype)
    for i in range(attns.shape[0]):
        attn = attns[i, :, 0, :CL, :CL]
        attn_mean = attn.mean(dim=0).detach().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(attn_mean, cmap='viridis')
        plt.colorbar()
        plt.title(f'Attention Weights Layer {i+1} {modeltype} Model')

        # Annotate each square with its value
        for y in range(attn_mean.shape[0]):
            for x in range(attn_mean.shape[1]):
                if x>y:
                    continue
                plt.text(x, y, f'{attn_mean[y, x]:.2f}', ha='center', va='center', color='white')
        # make xticks and yticks
        plt.xticks(range(CL), labels = [f'{i}' for i in range(CL)])
        plt.yticks(range(CL), labels = [f'{i}' for i in range(CL)])
        plt.ylabel('Neuron Number')
        plt.xlabel('Attending to Neuron Number')
        plt.show()


def matrix_formulation():
    key = 'underdamped'
    data = torch.load('data/dampedspring_data.pth')
    gammas = data[f'gammas_test_{key}']
    omegas0 = data[f'omegas_test_{key}']
    omegas = torch.sqrt(omegas0**2 - gammas**2)
    sequences = data[f'sequences_test_{key}']
    times = data[f'times_test_{key}']
    deltat = times[:,1] - times[:,0]
    # w00 = (cos + gamma/omegas*sin) * prefactor
        # w01 = (sin/omegas) * prefactor
        # w10 = (beta * sin) * prefactor
        # w11 = (cos - gamma/omegas*sin) * prefactor
    prefactor = torch.exp(-gammas*deltat)
    beta = - (gammas**2 + omegas**2)/omegas
    cos = torch.cos(omegas*deltat)
    sin = torch.sin(omegas*deltat)
    w00 = cos + gammas/omegas*sin
    w01 = sin/omegas
    w10 = beta * sin
    w11 = cos - gammas/omegas*sin
    terms = [w00, w01, w10, w11]
    matrix = torch.zeros((sequences.shape[0], 2, 2))
    for i in range(2):
        for j in range(2):
            matrix[:, i,j] = terms[i*2+j] * prefactor
    X,y = sequences[:,:-1, :], sequences[:, 1:, :]
    matrix = matrix.unsqueeze(1).repeat(1, X.shape[1], 1, 1) 
    pred = torch.einsum('btik,btk->bti', matrix, X)
    criterion = torch.nn.MSELoss()
    loss = criterion(pred, y)
    print(loss)
    return loss


def local_truncation_error(modeltype = 'underdamped', dataset = 'underdamped', datatype = 'train',neuron = 0):
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    data = torch.load('data/dampedspring_data.pth')
    sequences = data[f'sequences_{datatype}_{dataset}']
    times = data[f'times_{datatype}_{dataset}']
    gammas = data[f'gammas_{datatype}_{dataset}']
    omegas = data[f'omegas_{datatype}_{dataset}']
    X,y = sequences[:,:-1,:], sequences[:,1:, :]
    deltat = times[:,1] - times[:,0]
    ypred = model(X)
    lte = (ypred-y).abs()
    print(lte.shape)
    print(deltat.shape)
    fig, axs = plt.subplots(1,2, figsize = (10,5))
    colors = 'rb'
    
    for i, target in enumerate(['x','v']):
        nosmalldeltat = deltat>10**-3
        ax = axs[i]
        cax = ax.scatter(deltat[nosmalldeltat], lte[:,neuron, i][nosmalldeltat].detach().numpy(), c = (omegas-gammas)[nosmalldeltat], cmap='viridis')
        ax.set_xlabel('deltat (h)')
        ax.set_ylabel('Local Truncation Error (|ypred-y|)')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title(f'{target}')
    cbar = fig.colorbar(cax, ax=axs[-1])
    fig.suptitle(f'LTE vs h for Neuron {neuron}')
    plt.show()

if __name__ == '__main__':
    #compare_transformer_euler()
    data = torch.load('data/dampedspring_data.pth')
    gammas = data['gammas_test_damped']
    omegas = data['omegas_test_damped']
    sequences = data['sequences_test_damped']
    times = data['times_test_damped']
    deltat = times[:,1] - times[:,0]
    #matrix_formulation()
    #plot_R2evolve_euler(R2 = False)
    plot_R2evolve_euler(testtype = 'matrix_weights', R2 = False, layer = 2, neuronall = True)
    #local_truncation_error(neuron = 32)
    #probe_allneurons_target(testtype = 'matrix_weights', neuronall = True)
    #eulers_method(sequences, omegas, gammas, deltat, order = 2)
    #compare_transformer_euler(order =  2)
    #probetargets(plot = True)
    #plot_R2_firsths_euler()
    #plot_R2evolve_euler(R2 = False, CL = 65)
    #plot_R2evolve_euler()
    #linear_eulermapping()
    #plot_R2evolve_euler(R2 = False, novar = False)
    #compare_euler_model(run = False, CL = 65)
    
    #predict_allneurons_euler(target = 'v', epochs = 20000)
    #predict_neuron_euler()
    #plot_attention_maps()
    #plot_attention_maps(modeltype='', CL=10)


    # for targetname in ['x3x']:#['x0x','x1v','x2x','x2v','x3x','x3v','x4x','x4v']:
    #     plot_R2euler_datatypes(targetname = targetname)   
         
    #plot_R2euler_datatypes()
   
    
