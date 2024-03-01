import torch
from torch.utils.data import DataLoader, TensorDataset
from model import Transformer
from util import set_seed
from config import get_default_config
from tqdm import tqdm
from data import generate_springdata

def train(traindata, testdata,CL=10, fname = 'spring', dir = 'models', batch_size = 64, num_epochs = 20000, lr = 0.001):
    set_seed(0)
    config = get_default_config() 
    filebase = f'{fname}_{config.n_embd}emb_{config.n_layer}layer_{CL}CL_{num_epochs}epochs_{lr}lr_{batch_size}batch'
    print(filebase)
    modelfile = f'{dir}/{filebase}_model.pth'
    lossesfile = f'{dir}/{filebase}_losses.pth'

    traindata = traindata[:,:CL+1,:] # only use 10 timesteps for transformer predictions. it's shown an ability to learn off of this.
    X, y = traindata[:,:-1,:], traindata[:,1:,:]
    
    div = int(0.8*len(X))
    X_train, y_train = X[:div], y[:div]
    X_test_in, y_test_in = X[div:], y[div:]

    testdata = testdata[:,:CL+1,:] # only use 10 timesteps for transformer predictions.rest of data is for icl experiments
    print(testdata.shape)
    X_test_out, y_test_out = testdata[:,:-1,:], testdata[:,1:,:]


    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_in_dataset = TensorDataset(X_test_in, y_test_in)
    test_out_dataset = TensorDataset(X_test_out, y_test_out)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_in_loader = DataLoader(test_in_dataset, batch_size=batch_size, shuffle=False)
    test_out_loader = DataLoader(test_out_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = Transformer(config)
    model.train()

    # Loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Track the best model
    best_loss = float('inf')
    best_model_state = None

    # Training loop
    epoch_pbar = tqdm(range(num_epochs), desc='Training Progress')
    train_losses = []
    test_in_losses = []
    test_out_losses = []
    for epoch in epoch_pbar:
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Calculate train loss for the epoch
        epoch_train_loss = total_train_loss / len(train_loader)
        

        # Calculate test loss
        model.eval()
        total_test_in_loss = 0
        total_test_out_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_in_loader:
                output = model(batch_X)
                loss = criterion(output, batch_y)
                total_test_in_loss += loss.item()
            for batch_X, batch_y in test_out_loader:
                output = model(batch_X)
                loss = criterion(output, batch_y)
                total_test_out_loss += loss.item()

        # Calculate test loss for the epoch
        epoch_test_in_loss = total_test_in_loss / len(test_in_loader)
        epoch_test_out_loss = total_test_out_loss / len(test_out_loader)

        if epoch % 10 == 0:
            train_losses.append(epoch_train_loss)
            test_in_losses.append(epoch_test_in_loss)
            test_out_losses.append(epoch_test_out_loss)
        if epoch % 100 == 0:    
            torch.save({'train_losses': train_losses, 'test_in_losses': test_in_losses, 'test_out_losses': test_out_losses}, lossesfile)

        # Update the best model if the current loss is lower
        if epoch_test_in_loss < best_loss:
            best_loss = epoch_test_in_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, modelfile)
            
        # Update progress bar
        epoch_pbar.set_description(f'Epoch {epoch + 1}/{num_epochs}')
        epoch_pbar.set_postfix({'Train Loss': f'{epoch_train_loss:.2e}',
                        'Test-In Loss': f'{epoch_test_in_loss:.2e}',
                        'Test-Out Loss': f'{epoch_test_out_loss:.2e}'})

    # Save the best model
    torch.save(best_model_state, modelfile)
    torch.save({'train_losses': train_losses, 'test_in_losses': test_in_losses, 'test_out_losses': test_out_losses}, lossesfile)
    
    return model

if __name__ == '__main__':
    #generate_springdata(num_samples = 1000, sequence_length=50, plot = False)
    datadict = torch.load('data/spring_data.pth')
    traindata = datadict['sequences_train']
    trainomegas = datadict['train_omegas']
    testdata = datadict['sequences_test']
    testomegas = datadict['test_omegas']

    trained_model = train(traindata,testdata, CL = 10)
