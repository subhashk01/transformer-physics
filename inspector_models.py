import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from util import load_model, get_hidden_state_old, get_hidden_state
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import numpy as np
import os


class NonLinearProbe(nn.Module):
    def __init__(self, input_size, output_size=1):
        super().__init__()
        self.interim_size = input_size * 4
        self.l1 = nn.Linear(input_size, self.interim_size)
        self.l2 = nn.Linear(self.interim_size, output_size)
        # num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print(f"Number of parameters: {num_parameters}")

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x
    
class LinearProbe(nn.Module):
    def __init__(self, input_size, output_size=1):
        super().__init__()
        self.interim_size = input_size * 4
        self.l1 = nn.Linear(input_size, self.interim_size)
        self.l2 = nn.Linear(self.interim_size, output_size)
        # num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print(f"Number of parameters: {num_parameters}")

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x
    
class EllipseProbe(nn.Module):
    def __init__(self, poly_deg = 2):
        super().__init__()
        self.coeffs = nn.Parameter(torch.ones(poly_deg+1))
        self.poly_deg = poly_deg
        self.b = nn.Parameter(torch.tensor(3.0))  # Make b a trainable parameter

    def forward(self, pcs, omegas):
        # pcs is a 2D tensor of shape (N, 2)
        x, y = pcs[:, 0], pcs[:, 1]

        # Compute a(omega) for each omega
        a_omega = torch.zeros_like(omegas)
        for i in range(self.poly_deg + 1):
            a_omega += self.coeffs[i] * omegas.pow(i)

        # Calculate the left-hand side of the ellipse equation
        lhs = (x / a_omega).pow(2) + (y / self.b).pow(2)

        return lhs
    def loss(self, pcs, omegas):
        # The target value is 1 for all points (since it's an ellipse equation)
        target = torch.ones_like(pcs[:, 0])
        lhs = self.forward(pcs, omegas)
        return torch.mean((lhs - target).pow(2))  # Mean squared error
    


        
    

def train_probe(inspector, train_hidden_states, train_target_vals, test_hidden_states, test_target_vals, epochs=10000, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(inspector.parameters(), lr=learning_rate)
    #train_losses, test_losses = [], []

    epoch_pbar = tqdm(range(epochs), desc='Training Probe')

    for epoch in epoch_pbar:
        inspector.train()
        optimizer.zero_grad()
        outputs = inspector(train_hidden_states)
        loss = criterion(outputs.squeeze(), train_target_vals)

        loss.backward()
        optimizer.step()
        #train_losses.append(loss.item())

        with torch.no_grad():
            inspector.eval()
            test_outputs = inspector(test_hidden_states)
            test_loss = criterion(test_outputs.squeeze(), test_target_vals)
            #test_losses.append(test_loss.item())

        epoch_pbar.set_description(f'Epoch {epoch}')
        epoch_pbar.set_postfix({'Train Loss': f'{loss.item():.2e}',
                        'Test Loss': f'{test_loss.item():.2e}'})

    return loss.item(), test_loss.item()

def probe_hiddenstate(model, modelname, data, target_name, target_vals, linear = False, layer = 0, neuron = 0, CL = 10, epochs = 10000, plot = True):

    hidden_states= get_hidden_state(model, data, CL, layer, neuron)

    linear_name = {True: 'Linear', False: 'NonLinear'}
    if linear:
        inspector = LinearProbe(hidden_states.shape[-1], 1)
    else:
        inspector = NonLinearProbe(hidden_states.shape[-1], 1)

    # randomize hidden_states and target_vals
    indices = torch.randperm(hidden_states.shape[0])
    hidden_states = hidden_states[indices]
    target_vals = target_vals[indices] #TODO CHANGE BACK
    # split into test train
    div = int(0.8*hidden_states.shape[0])
    train_hidden_states = hidden_states[:div]
    test_hidden_states = hidden_states[div:]
    train_target_vals = target_vals[:div]
    test_target_vals = target_vals[div:]

    train_loss, test_loss = train_probe(inspector, train_hidden_states, train_target_vals, test_hidden_states, test_target_vals, epochs = epochs)
    target_pred = inspector(hidden_states).detach().numpy()
    r_squared = r2_score(target_vals, target_pred)

    # see if folder model name is in directory probes
    if not os.path.exists(f'probes/{modelname}'):
        os.makedirs(f'probes/{modelname}')
    # save inspector model
    torch.save(inspector.state_dict(), f'probes/{modelname}/{target_name}_layer{layer}_neuron{neuron}_{linear_name[linear]}_probe.pth')
    if plot:
        plt.scatter(target_vals, target_pred, color = 'b')
        plt.title(f'2Layer NN Predictor of {target_name} from Hidden State\nLayer = {layer}, Neuron = {neuron}\nR^2: {r_squared:.2f}. Train Loss = {train_loss:.2e}, Test Loss = {test_loss:.2e}')
        # linear regression between target vals and pred
        plt.xlabel(f'True Values of {target_name}')
        plt.ylabel(f'Predicted Values of {target_name}')
        plt.show()

    return r_squared

def probe_hiddenstates(model, modelname, data, target_vals, target_name, linear = False, CL = 10, epochs = 10000, num_layers = 2):
    correlations = []
    for layer in range(num_layers+1):
        layer_corr = []
        for neuron in range(CL):
            print(f'Layer: {layer}, Neuron: {neuron}')
            r2 = probe_hiddenstate(model, modelname, data, layer = layer, neuron = neuron, target_vals = target_vals, target_name = target_name,  CL = CL, epochs = epochs, plot = False, linear = linear)
            print(f'R^2: {r2:.2f}')
            layer_corr.append(r2)
        correlations.append(layer_corr)
    correlations = np.array(correlations)

    layer_corr = np.mean(correlations, axis=1)  # Average correlation for each layer
    hs_corr = np.mean(correlations, axis=0)  # Average correlation for each hidden state
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 5))
    cax = ax.matshow(correlations, cmap='viridis')
    cbar = fig.colorbar(cax)
    labeldict = {True: 'Linear', False: 'Non Linear'}
    cbar.set_label(f'{labeldict[linear]} R^2')  # Set label for color bar

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

    ax.set_title(f'{modelname} Model \n R^2 of {labeldict[linear]} 2Layer NN w/ Hidden States with {target_name}\nTest on CL = {CL}, Train w/ {num_layers}Layers, 16Emb, 65CL')
    plt.savefig(f'figures/{modelname}_{target_name}_{labeldict[linear]}_probes.png')
    return np.array(correlations)  # Shape: [3, 10], representing overall correlation for each hidden state in each layer



def check_ellipse(model, layer=0, neuron=0, target='omegas', CL=10, poly_deg = 2):
    hidden_states, target_vals = get_hidden_state_old(model, CL, layer, neuron, target)
    ellipse_probe = EllipseProbe(poly_deg=poly_deg)
    optimizer = optim.Adam(ellipse_probe.parameters(), lr=0.01)
    print('here1')
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(hidden_states)
    print('here2')
    pcs = torch.tensor(pcs, dtype=torch.float32)

    pbar = tqdm(range(10000), desc='Initializing')
    for epoch in pbar:
        optimizer.zero_grad()
        loss = ellipse_probe.loss(pcs, target_vals)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == len(pbar) - 1:
            pbar.set_description(f'Epoch: {epoch}, Loss: {loss.item():.2e}, Coeffs: {[f"{i.item():.2f}" for i in ellipse_probe.coeffs]}, b: {ellipse_probe.b.item():.4f}')


    # # After training, you can access the learned parameters
    # print('Learned coefficients:', ellipse_probe.coeffs.data)
    # print('Learned b:', ellipse_probe.b.item())

    # target_pred = ellipse_probe(hidden_states, target_vals).detach().numpy()

    # plt.scatter(target_vals, target_pred, color = 'b')
    # r_squared = r2_score(target_vals, target_pred)
    # title = f'Ellipse Predictor of Omega from Hidden State\nLayer = {layer}, Neuron = {neuron}\nR^2: {r_squared:.2f}. MSELoss = {loss.item():.2e}'
    # form = r'$\frac{x^2}{a(\omega)^2} + \frac{y^2}{b^2} = 1, a(\omega) = \sum_{i=0}^{2} c_i \omega^i$'
    # coeffs = [f"{i.item():.2f}" for i in ellipse_probe.coeffs]
    # plt.title(title + '\n' + form + f'\nCoeffs: {coeffs}, b: {ellipse_probe.b.item():.4f}')
    # plt.show()
    # linear regression between target vals and pred
            
# def plot_ellipse(model, layer=0, neuron=0, target='omegas', CL=10):
#     hidden_states, target_vals = get_hidden_state(model, CL, layer, neuron, target)
#     pca = PCA(n_components=2)
#     pcs = pca.fit_transform(hidden_states)
#     pcs = torch.tensor(pcs, dtype=torch.float32)
#     xs,ys = pcs[:, 0], pcs[:, 1]
#     fig, ax = plt.subplots(figsize = (10,6))
#     cax = ax.scatter(pcs[:, 0], pcs[:, 1], c=target_vals, cmap='viridis')
#     cbar = fig.colorbar(cax)
#     xran = np.linspace(xs.min(), xs.max(), 100)
#     yran = np.linspace(-2, 2, 100)
#     ws = []
#     for x in xran:
#         for y in yran:
#             w = np.sqrt(x**2/ (1.4**2 - y**2/4))
#             ws.append(w)

def plot_ellipse(model, layer=0, neuron=0, target='omegas', CL=10):
    hidden_states, target_vals = get_hidden_state_old(model, CL, layer, neuron, target)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(hidden_states)
    pcs = torch.tensor(pcs, dtype=torch.float32)
    xs, ys = pcs[:, 0], pcs[:, 1]

    fig, ax = plt.subplots(figsize=(10, 9))

    # Create the meshgrid for the background
    xran = np.linspace(xs.min(), xs.max(), 100)
    yran = np.linspace(ys.min(), ys.max(), 100)
    X, Y = np.meshgrid(xran, yran)
    W = np.zeros_like(X)

    def calculate_w(x, y):
        w = np.sqrt(x**2 / (1.4**2 - y**2 / 4))
        return w
    W = calculate_w(X, Y)
    criterion = nn.MSELoss()
    loss = criterion(calculate_w(xs,ys), target_vals)

    # Plot the w values as a colored mesh
    # Use the same colormap and normalization as the scatter plot
    norm = plt.Normalize(target_vals.min(), target_vals.max())
    ax.pcolormesh(X, Y, W, cmap='viridis', norm=norm, shading='auto', alpha=0.2, label = 'Ellipse Mapping')

    # Plot the scatter points on top of the colored mesh
    cax = ax.scatter(xs, ys, c=target_vals, cmap='viridis', norm=norm, label = 'Hidden States')
    cbar = fig.colorbar(cax, ax=ax)
    title = f'Ellipse Predictor of {target} from Hidden State\nLayer = 0, Neuron = 0\nMSELoss = {loss.item():.2e}\n'
    eq = r'$\frac{1}{1.4^2}(\frac{PC1^2}{\omega^2} + \frac{PC2^2}{2^2}) = 1$'
    plt.title(title+eq)
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 {var[0]*100:.2f}% Var')

    ax.set_ylabel(f'PC2 {var[1]*100:.2f}% Var')
    cbar.set_label(f'{target}')
    plt.legend()
    plt.show()

    # Second plot with color bar for xs values
    plt.figure(figsize=(10, 9))
    preds = calculate_w(xs,ys)
    ys = ys.abs()
    r2 = r2_score(target_vals, preds)
    norm = plt.Normalize(ys.min(), ys.max())
    plt.scatter(target_vals, preds, c=ys, cmap='viridis', norm=norm)
    plt.ylabel('Predicted Values')
    plt.xlabel('True Values')
    plt.title(f'Predicted vs True Values of {target} from Hidden State\nLayer = 0, Neuron = 0\nMSELoss = {loss.item():.2e}, R^2 = {r2:.2f}')
    cbar = plt.colorbar()
    cbar.set_label('|PC2 values|')
    plt.show()


if __name__ == '__main__':
    losspath = 'models/spring_16emb_2layer_65CL_20000epochs_0.001lr_64batch_losses.pth'
    modelpath = 'models/spring_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth'
    model = load_model(file = modelpath)
    #check_ellipse(model, poly_deg = 1)
    #plot_ellipse(model)
    #probe_hiddenstate(model, layer = 2, neuron = 0, target = 'omegas')
    probe_hiddenstates(model, target = 'omegas', CL = 10, epochs = 10000, num_layers = 2, linear = True)