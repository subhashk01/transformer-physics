import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from util import load_model
import matplotlib.pyplot as plt
from inspector_models import LinearProbe, LinearDirect
from util import get_hidden_state, euler_to_term
from sklearn.metrics import r2_score
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from tqdm import tqdm


def cosine_similarity(sequence1, sequence2):
    # Normalize the sequences along the last dimension
    sequence1_norm = F.normalize(sequence1, p=2, dim=-1)
    sequence2_norm = F.normalize(sequence2, p=2, dim=-1)
    
    # Compute the dot product between the normalized sequences
    # Sum along the last dimension to get the cosine similarity for each time step
    cos_sim = torch.sum(sequence1_norm * sequence2_norm, dim=-1)
    
    return cos_sim



def eulers_method(seqin, omegas, gammas, deltat, order=1):
    # given some array with last dimension size 2
    # euler update and give me an array of the same shape

    # zero order is identity matrix
    

    sequences = seqin[:,:11,:]
    num = sequences.shape[0]
    CL = sequences.shape[1]-1


    zero_order = torch.zeros(num, CL, 2, 2)
    zero_order[:,:,0,0] = 1
    zero_order[:,:, 1,1] = 1

    readin1 = torch.load('probes_underdamped_65_omega0^2deltat_deltat_gammadeltat.pth')
    first_order = torch.zeros(num, CL, 2, 2)   
    first_order[:,:, 0,1] = readin1['deltat'][:,-1,:]
    first_order[:, :, 1, 0] = -readin1['omega0^2deltat'][:,-1,:]
    first_order[:, :, 1, 1] = -2*readin1['gammadeltat'][:,-1,:]

    readin2 = torch.load('probes_underdamped_65_fourgamma^2minusomega0^2deltat^2_omega0^2deltat^2_gammadeltat^2_gammaomega0^2deltat^2.pth')
    second_order = torch.zeros(num, CL, 2, 2)
    second_order[:,:,0,0] = -0.5 * readin2['omega0^2deltat^2'][:,-1,:]
    second_order[:,:,0,1] = - readin2['gammadeltat^2'][:,-1,:]
    second_order[:,:,1,0] = readin2['gammaomega0^2deltat^2'][:,-1,:]
    second_order[:,:,1,1] = 0.5*readin2['fourgamma^2minusomega0^2deltat^2'][:,-1,:]


    euler_matrices = [zero_order, first_order, second_order]
    euler_matrix = torch.zeros(zero_order.shape)
    for i in range(order+1):
        euler_matrix = euler_matrix + euler_matrices[i]

    X,y = sequences[:,:-1, :], sequences[:, 1:, :]
    #euler_matrix = euler_matrix.unsqueeze(1).repeat(1, X.shape[1], 1, 1)
    y_euler = torch.einsum('btik,btk->bti', euler_matrix, X)

    # euler_sequences = (zero_order + first_order) 
    # print(sequences.shape)
    # print(euler_sequences.shape)

    criterion = torch.nn.MSELoss()
    loss = criterion(y_euler, y)
    print(loss)
    return y_euler
    
def compare_transformer_euler(order = 1):
    # comparing the underdamped model
    data = torch.load('data/dampedspring_data.pth')
    gammas = data['gammas_test_damped']
    omegas = data['omegas_test_damped']
    sequences = data['sequences_test_damped'][:,:11,:]
    times = data['times_test_damped']
    deltat = times[:,1] - times[:,0]
    
    euler_y = eulers_method(sequences, omegas, gammas, deltat, order = order)
    X,y = sequences[:,:-1, :], sequences[:, 1:, :]
    model = load_model('models/springunderdamped_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    model.eval()
    y_model = model(X)
    MSE_model = (y_model - y).pow(2)
    MSE_euler = (euler_y - y).pow(2)
    cos_simsx = cosine_similarity(MSE_model[:,:,0], MSE_euler[:,:,0])
    cos_simsv = cosine_similarity(MSE_model[:,:,1], MSE_euler[:,:,1])
    cossimx = cos_simsx.mean()
    cossimv = cos_simsv.mean()
    for i in range(MSE_model.shape[0]):
        MSE_modeltraj, MSE_eulertraj = MSE_model[i].detach().numpy(), MSE_euler[i].detach().numpy()
        MSE_modeltrajx, MSE_modeltrajv = MSE_modeltraj[:,0], MSE_modeltraj[:,1]
        MSE_eulertrajx, MSE_eulertrajv = MSE_eulertraj[:,0], MSE_eulertraj[:,1]
        print(MSE_eulertrajx)
        print(MSE_modeltrajx)

        print(MSE_eulertrajv)
        print(MSE_modeltrajv)
        
        datanum = list(range(MSE_modeltraj.shape[0]))
        plt.plot(datanum, MSE_modeltraj.mean(axis=1), color = 'b',label = 'Transformer Model')
        plt.plot(datanum, MSE_eulertraj.mean(axis=1), color = 'r',label = 'Euler Method')
        # plt.plot(datanum, MSE_modeltrajx, color = 'b',label = 'Transformer Model - MSE x')
        # plt.plot(datanum, MSE_eulertrajx, color = 'b',label = 'Euler Method - MSE x', linestyle = '--')
        # plt.plot(datanum, MSE_modeltrajv, color = 'r',label = 'Transformer Model - MSE v')
        # plt.plot(datanum, MSE_eulertrajv, color = 'r',label = 'Euler Method - MSE v', linestyle = '--')
        #plt.plot(datanum, MSE_model_eulertraj.detach().numpy(), color = 'g', linestyle = '--', label = 'Delta Between Transformer, Euler')
        plt.xlabel('Timestep')
        plt.ylabel('MSE from True Trajectory')
        plt.yscale('log')
        plt.title(f'Trajectory {i}\nComparing Transformer and Euler Method (Order = {order}) Predictions\n Cosine Sim Avg = {cossimx:.2f}, this traj = {cos_simsx[i].item():.2f}')
        plt.legend()
        plt.show()
        if i == 3:
            break

    # print(loss_model, loss_euler)
    # print(y_model.shape)
    # print(euler_y.shape)


def probetargets(modeltype = 'underdamped',CL = 65, layer = 2, plot = False):
    data = torch.load('data/dampedspring_data.pth')
    gammas = data['gammas_test_damped']
    omegas = data['omegas_test_damped']
    sequences = data['sequences_test_damped']
    times = data['times_test_damped']
    deltat = times[:,1] - times[:,0]
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    model.eval()
    X,y = sequences[:,:-1, :], sequences[:, 1:, :]
    #targetdict = {'omega0^2deltat': omegas**2 * deltat, 'deltat': deltat, 'gammadeltat': gammas*deltat}
    targetdict = {'fourgamma^2minusomega0^2deltat^2': (4*gammas**2 - omegas**2) * deltat**2, 
                  'omega0^2deltat^2': omegas**2 * deltat**2, 
                  'gammadeltat^2': gammas*deltat**2,
                  'gammaomega0^2deltat^2': gammas*omegas**2*deltat**2}
    predicteddict = {}
    criterion = torch.nn.MSELoss()
    savestr = f'probes_{modeltype}_{CL}'
    _, hss = model.forward_hs(X)
    hss = torch.stack(hss)
    hss = hss.view(hss.shape[0], hss.shape[-2], hss.shape[1], hss.shape[-1])
    for targetname, target in targetdict.items():
        layer_vals = []
        for layer in range(3):
            neuron_vals = []
            for neuron in range(10):
                hs = get_hidden_state(model, X, CL = CL, layer = layer, neuron = neuron)
                #hs = hss[layer,neuron,:,:]
                probe = LinearProbe(hs.shape[1])
                probepath = f'probes/{modeltype}/{targetname}_layer{layer}_neuron{neuron}_Linear_probe.pth'
                probe.load_state_dict(torch.load(probepath))
                target_pred = probe(hs).squeeze()
                loss = criterion(target_pred, target)
                neuron_vals.append(target_pred)
                r2 = r2_score(target.detach().numpy(), target_pred.detach().numpy())
                if plot and layer == 2 and neuron == 0: 
                    plt.scatter(target.detach().numpy(), target_pred.detach().numpy(),color = 'b')
                    plt.title(f'Probe on {targetname}\nUnderdamped model, Layer {layer}, Neuron {neuron}\nMSE = {loss.item():.2e}, R^2 = {r2:.2f}')
                    breakpoint()
                    plt.show()
                    
                print(f'{targetname}: Layer {layer}, Neuron {neuron}, MSE = {loss.item():.2e}, R^2 = {r2:.2f}')
                
            neuron_vals = torch.stack(neuron_vals, dim = 1)
            layer_vals.append(neuron_vals)
        layer_vals = torch.stack(layer_vals, dim = 1)
        predicteddict[targetname] = layer_vals
        savestr += f'_{targetname}'
    # save predict dict
    print(savestr)
        
    torch.save(predicteddict, savestr+'.pth')

def plot_R2_firsths_euler(modeltype = 'underdamped'):
    plt.rcParams['text.usetex'] = True
    layer, neuron = 2,0
    data = torch.load('data/dampedspring_data.pth')
    data = torch.cat((data[f'sequences_test_damped'], data[f'sequences_train_damped']))
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    model.eval()
    target_dict = torch.load('euler_terms_underdamped.pth')
    colors = 'brgcmyk'
    plt.figure(figsize = (10,5))
    for e, euler_target in enumerate(['x','v']):
        avg_mags = []
        R2s = []
        targetnames = []
        for targetname in [f'{euler_target}{i}{term}' for i in range(5) for term in ['x','v']]:
            if targetname in target_dict.keys():
                target = target_dict[targetname][:,neuron]
            else:
                continue
            hs = get_hidden_state(model, data, CL = 1, layer = layer, neuron = neuron)
            probe = LinearProbe(hs.shape[1])
            probepath = f'probes/{modeltype}/{targetname}_layer{layer}_neuron{neuron}_Linear_probe.pth'
            probe.load_state_dict(torch.load(probepath))
            target_pred = probe(hs).squeeze()
            r2 = r2_score(target.detach().numpy(), target_pred.detach().numpy())
            R2s.append(r2)
            avg_mags.append(target.abs().mean())
            targetnames.append(targetname)
        #threshold all R2s values below 0 to 0
        R2s_adj = [max(0,r2) for r2 in R2s]
        plt.scatter(avg_mags, R2s_adj, color = colors[e], label = rf'${euler_target}(t+dt)$ Taylor Expansion Terms')
        # add text to scatter points with targetnames
        for i, txt in enumerate(targetnames):
            plt.annotate(txt, (avg_mags[i]*1.1, R2s_adj[i]))
    plt.xlabel('Average Magnitude of Term')
    plt.xscale('log')
    plt.ylabel(r'$R^2$ of Linear Probe of Term')
    plt.legend()
    plt.title(rf'$R^2$ of Linear Probes for Taylor Expansion\\{modeltype} Model, Layer {layer}, Neuron {neuron}')
    plt.show()


def plot_R2evolve_euler(modeltype = 'underdamped',CL = 65, layer = 2, R2 = True, novar = False):
    # if R2 is true, plot R2. else plot MSE
    plt.rcParams['text.usetex'] = True
    data = torch.load('data/dampedspring_data.pth')
    data = torch.cat((data[f'sequences_test_damped'], data[f'sequences_train_damped']))
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    model.eval()
    if novar:
        target_dict = torch.load('euler_termsnovar_underdamped.pth')
    else:
        target_dict = torch.load('euler_terms_underdamped.pth')
    colors = 'br'
    plt.figure(figsize = (10,5))
    layer = 2
    mapping = euler_to_term(novar)
    # Create a normalization for the color mapping
    norm = LogNorm(vmin=1e-5, vmax=1e0)
    str = '$R^2$' if R2 else 'MSE'
    criterion = torch.nn.MSELoss()
    for e, euler_target in enumerate(['x','v']):
        fig, axs = plt.subplots(5,2, sharey = R2, sharex = R2, figsize = (10,10))
        term_num = 0
        for targetname in [f'{euler_target}{i}{term}' for i in range(5) for term in ['x','v']]:
            col_num = term_num % 2
            row_num = term_num // 2
            ax = axs[row_num, col_num]
            if col_num==0:
                ax.set_ylabel(rf'Order {row_num} Terms\\{str}')
            if row_num == axs.shape[0]-1:
                ax.set_xlabel('Neuron Number')
            if row_num == 0:
                if col_num == 0:
                    ax.set_title(rf'$x$ terms in Taylor Expansion of ${euler_target}(t+dt)$')
                else:
                    ax.set_title(rf'$v$ terms in Taylor Expansion of ${euler_target}(t+dt)$')
            neurons = list(range(10))
            R2s = []
            avg_mags = []
            MSEs = []
            for neuron in neurons:
                if targetname in target_dict.keys():
                    target = target_dict[targetname][:,neuron]
                else:
                    continue
                hs = get_hidden_state(model, data, CL = neuron+1, layer = layer, neuron = neuron)
                probe = LinearProbe(hs.shape[1])
                probepath = f'probes/{modeltype}/{targetname}_layer{layer}_neuron{neuron}_Linear_probe.pth'
                probe.load_state_dict(torch.load(probepath))
                target_pred = probe(hs).squeeze()
                r2 = r2_score(target.detach().numpy(), target_pred.detach().numpy())
                R2s.append(r2)
                avg_mags.append(target.abs().mean())
                mse = criterion(target_pred, target).detach()
                MSEs.append(mse)
            
            if len(R2s):
                print(targetname, row_num, col_num)
                R2s = [max(0,r2) for r2 in R2s]
                metric = R2 if R2 else MSEs
                ax.plot(neurons, metric, color = colors[e], marker = 'o', label = rf'${targetname}:{mapping[targetname]}$\\Avg Mag = {target.abs().mean().item():.2e}')

                #scatter = ax.scatter(neurons, R2s, c=avg_mags, cmap=cm.viridis, norm=norm, marker='o', label=targetname)
                ax.legend()

            term_num+=1
            if R2:
                ax.set_ylim(-0.2,1.2)
            else:
                ax.set_xscale('log')
                ax.set_yscale('log')
        fig.suptitle(rf'Evolution of {str} of Linear Probes for Taylor Expansion ${euler_target}(t+dt)$\\{modeltype} Model, Layer {layer}')
        # cbar = fig.colorbar(scatter, ax=axs.ravel().tolist(), orientation='vertical')
        # cbar.set_label('Average Magnitude (log scale)')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        
def plot_R2euler_datatypes(targetname='x0x', layer=2, neuron=0, modeltype='underdamped', CL=65):
    plt.rcParams['text.usetex'] = True
    mapping = euler_to_term()
    data = torch.load('data/dampedspring_data.pth')
    omega0 = torch.cat((data[f'omegas_test_damped'], data[f'omegas_train_damped']), dim=0).unsqueeze(1)
    gamma = torch.cat((data[f'gammas_test_damped'], data[f'gammas_train_damped']), dim=0).unsqueeze(1)
    data = torch.cat((data[f'sequences_test_damped'], data[f'sequences_train_damped']))
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    model.eval()
    target_dict = torch.load('euler_terms_underdamped.pth')
    plt.figure(figsize=(10, 7))
    if targetname in target_dict.keys():
        target = target_dict[targetname][:, neuron]
    else:
        assert 'Invalid targetname'
    hs = get_hidden_state(model, data, CL=neuron + 1, layer=layer, neuron=neuron)
    probe = LinearProbe(hs.shape[1])
    probepath = f'probes/{modeltype}/{targetname}_layer{layer}_neuron{neuron}_Linear_probe.pth'
    probe.load_state_dict(torch.load(probepath))
    target_pred = probe(hs).squeeze()
    # get all indices where omega0 > gamma
    diff = (omega0 - gamma).squeeze()
    underdamped_indices = (diff > 0).nonzero().squeeze()
    overdamped_indices = (diff < 0).nonzero().squeeze()



    r2_under = r2_score(target[underdamped_indices].detach().numpy(), target_pred[underdamped_indices].detach().numpy())
    r2_over = r2_score(target[overdamped_indices].detach().numpy(), target_pred[overdamped_indices].detach().numpy())

    # Create a color array based on the condition
    #colors = ['red' if o < g else 'blue' for o, g in zip(omega0, gamma)]
    colors = omega0 - gamma

    # Plot target_pred vs. target with conditional coloring
    plt.scatter(target.detach().numpy(), target_pred.detach().numpy(), c=colors, cmap='coolwarm', alpha = 0.5, label = rf'')
    #splt.colorbar(label='Omega0 - Gamma')
    plt.xlabel('Target')
    plt.ylabel('Predicted Target')
    #plt.title(rf'{modeltype} Generalization for Target: {targetname}\n Underdamped $R^2 = {r2_under:.2f}$, Overdamped $R^2 {r2_over:.2f}$ \n Layer: {layer} | Neuron: {neuron} | Model: {modeltype}')
    plt.title(rf'{modeltype} Generalization for Target: {targetname}, ${mapping[targetname]}$\\Underdamped $R^2 = {r2_under:.2f}$, Overdamped $R^2 = {r2_over:.2f}$ \\ Layer: {layer}, Neuron: {neuron}, Model: {modeltype}')
    plt.show()
    

def eulermapping_data(neuron=0, target = 'x'):
    target_dict = torch.load('euler_terms_underdamped.pth')
    inputs = []
    targetnames =[]
    coeffs = {'x0x': 1, 'x1v': 1, 'x2v': -1, 'x2x': -1/2, 'x3v': 1/6, 'x3x': 1/3, 'x4v': 1/6, 'x4x': 1/24,
              'v0v': 1, 'v1v': -2, 'v1x': -1, 'v2v': 0.5, 'v2x': 1, 'v3v': 2/3, 'v3x': 1/6, 'v4v': 1/24, 'v4x': 1/6}
    for targetname in target_dict:
        if targetname[0] == target:
            targetvals = coeffs[targetname]*target_dict[targetname][:,neuron]
            targetnames.append(targetname)
            inputs.append(targetvals)
    inputs = torch.stack(inputs, dim = 1)
    return inputs, targetnames

def predict_neuron_euler(layer=2, neuron=0, target='x', modeltype='underdamped', epochs=20000, lr=1e-3):
    _, targetnames = eulermapping_data(neuron, target)
    data = torch.load('data/dampedspring_data.pth')
    data = torch.cat((data[f'sequences_test_damped'], data[f'sequences_train_damped']))[:, :neuron + 2, :]
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    model.eval()
    outputs = model(data[:,:neuron+1, :])
    
    ind = {'x': 0, 'v': 1}
    output = outputs[:, neuron, ind[target]]
    real_output = data[:,neuron+1, ind[target]]
  
    criterion = nn.MSELoss()

    inputs = []
    for targetname in targetnames:
        hs = get_hidden_state(model, data, CL = neuron+1, layer = layer, neuron = neuron)
        probe = LinearProbe(hs.shape[1])
        probepath = f'probes/{modeltype}/{targetname}_layer{layer}_neuron{neuron}_Linear_probe.pth'
        probe.load_state_dict(torch.load(probepath))
        input = probe(hs).squeeze()
        inputs.append(input)
    inputs = torch.stack(inputs, dim = 1)

        

    # Randomize the order of the inputs
    indices = torch.randperm(inputs.shape[0])
    inputs = inputs[indices]
    output = output[indices]
    real_output = real_output[indices]

    div = int(0.8 * inputs.shape[0])
    Xtrain = inputs[:div].detach()
    ytrain = output[:div].detach()
    Xtest = inputs[div:].detach()
    ytest = output[div:].detach()
    real_output_test = real_output[div:].detach()

    MSE_model_actual = criterion(real_output_test, ytest)
    MSE_probe_actual_best = float('inf')

    best_R2 = float('-inf')
    best_MSE = float('inf')
    best_probe = None

    for _ in range(1):
        probe = LinearDirect(inputs.shape[1])
        epoch_pbar = tqdm(range(epochs), desc='Training Probe')

        
        optimizer = optim.Adam(probe.parameters(), lr=lr)
        lambda_l1 = 1e-7  # Regularization strength

        for _ in epoch_pbar:
            probe.train()
            optimizer.zero_grad()
            ytrain_pred = probe(Xtrain)
            loss = criterion(ytrain_pred.squeeze(), ytrain)

            # Add L1 regularization
            l1_reg = 0
            for param in probe.parameters():
                l1_reg += param.abs().sum()
            loss += lambda_l1 * l1_reg

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probe.eval()
                ytest_pred = probe(Xtest)
                test_loss = criterion(ytest_pred.squeeze(), ytest)
                r2_test = r2_score(ytest.detach().numpy(), ytest_pred.detach().numpy())
                r2_train = r2_score(ytrain.detach().numpy(), ytrain_pred.detach().numpy())
                MSE_probe_actual = criterion(ytest_pred.squeeze(), real_output_test)
                if test_loss < best_MSE:
                    best_R2 = r2_test
                    best_MSE = test_loss
                    MSE_probe_actual_best = MSE_probe_actual
                    best_probe = probe.state_dict()



            epoch_pbar.set_postfix({'Train Loss': loss.item(), 'Test Loss': test_loss.item(), 'R^2 Train': r2_train, 'R^2 Test': r2_test})
        

        torch.save(best_probe, f'probes/eulerlinear_{target}target_{modeltype}_layer{layer}_neuron{neuron}_Linear_probe.pth')
        best_weights = probe.l.weight.squeeze()
        for i, targetname in enumerate(targetnames):
            print(f'{targetname}: {best_weights[i].item():.2f}')
    return best_MSE, best_R2, best_weights, MSE_probe_actual_best, MSE_model_actual

def compare_euler_model(run = True, modeltype = 'underdamped', num_neurons = 10):
    # THEORY: if euler was well approximating the model and not just data,
    # we expect MSE(model, euler) < MSE(euler, data), MSE(model, data)
    neurons = list(range(num_neurons))
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    model.eval()

    data = torch.load('data/dampedspring_data.pth')
    data = torch.cat((data[f'sequences_test_damped'], data[f'sequences_train_damped']))[:, :num_neurons + 2, :]
    outputs = model(data[:,:num_neurons+1, :])
    
    
    ind = {'x': 0, 'v': 1}
    criterion = nn.MSELoss()
    for target in ['x', 'v']:
        r2s = []
        MSEs_probe_model = []
        MSEs_probe_actual = []
        MSEs_model_actual = []
        for neuron in neurons:
            print(target, neuron)
            if run:
                MSE_probe_model, r2, _, MSE_probe_actual, MSE_model_actual = predict_neuron_euler(target = target, neuron = neuron, epochs = 10000)
            else:
                inputs, _ = eulermapping_data(neuron = neuron, target = target)

                output = outputs[:, neuron, ind[target]]
                r2 = r2_score(output.detach().numpy(), inputs[:,0].detach().numpy())
                print(f'{r2:.2f}')
                real_output = data[:,neuron+1, ind[target]]
                straight_euler = torch.sum(inputs, dim = 1)
                straight_euler = inputs[:,0]+1.1*inputs[:,1]
                MSE_probe_model = criterion(output, straight_euler).detach()
                MSE_probe_actual = criterion(straight_euler, real_output).detach()
                MSE_model_actual = criterion(output, real_output).detach()

            MSEs_probe_model.append(MSE_probe_model)
            MSEs_probe_actual.append(MSE_probe_actual)
            MSEs_model_actual.append(MSE_model_actual)
            r2s.append(r2s)
        plt.figure(figsize = (7,7))
        plt.plot(neurons, MSEs_probe_model, color = 'b',marker = 'o', label = 'MSE(euler probe, transformer)')
        plt.plot(neurons, MSEs_probe_actual, color = 'r', marker = 'o', label = 'MSE(euler probe, actual data)')
        plt.plot(neurons, MSEs_model_actual, color = 'g', marker = 'o', label = 'MSE(transformer, actual data)')
        plt.title(f'Euler Probe Approximation of Model vs Data for {target}')
        plt.legend()
        plt.ylabel('MSE')
        plt.yscale('log')
        plt.xlabel('Neuron')
        plt.savefig(f'figures/euler_model_comparison_{target}_{num_neurons}CL.png')
        #plt.show()


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




if __name__ == '__main__':
    #compare_transformer_euler()
    data = torch.load('data/dampedspring_data.pth')
    gammas = data['gammas_test_damped']
    omegas = data['omegas_test_damped']
    sequences = data['sequences_test_damped']
    times = data['times_test_damped']
    deltat = times[:,1] - times[:,0]
    #eulers_method(sequences, omegas, gammas, deltat, order = 2)
    #compare_transformer_euler(order =  2)
    #probetargets(plot = True)
    #plot_R2_firsths_euler()
    #plot_R2evolve_euler()
    #linear_eulermapping()
    #plot_R2evolve_euler(R2 = False, novar = False)
    compare_euler_model(run = True, num_neurons = 10)
    #predict_neuron_euler()
    #plot_attention_maps()
    


    # for targetname in ['x3x']:#['x0x','x1v','x2x','x2v','x3x','x3v','x4x','x4v']:
    #     plot_R2euler_datatypes(targetname = targetname)   
         
    #plot_R2euler_datatypes()
   
    
