import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from model import Transformer
from util import set_seed
from config import get_default_config, linreg_config
from tqdm import tqdm
from data import generate_springdata, omega1to2, generate_dampedspringdata
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def train(config, traindata, testdata,CL=65, loadmodel = False, fname = 'spring', dir = 'models', batch_size = 64, num_epochs = 20000, lr = 0.001):
    set_seed(10)
    if config is None:
        config = get_default_config() 
    base = f'{fname}_{config.n_embd}emb_{config.n_layer}layer'
    filebase = f'{base}_{CL}CL_{num_epochs}epochs_{lr}lr_{batch_size}batch'
    totalbase = f'{dir}/{base}/{filebase}'
    modelfile = f'{totalbase}_model.pth'
    lossesfile = f'{totalbase}_losses.pth'

    if base not in os.listdir(dir):
        os.mkdir(f'{dir}/{base}')

    if 'linreg' in fname:
        traindata = traindata.unsqueeze(-1)
        testdata = testdata.unsqueeze(-1)

    traindata = traindata[:,:CL+1,:] # only use 10 timesteps for transformer predictions. it's shown an ability to learn off of this.
    X, y = traindata[:,:-1,:], traindata[:,1:,:]
    #X, y = trainxy #TODO CHANGE IMPLEMENTATION
    
    div = int(0.8*len(X))
    X_train, y_train = X[:div], y[:div]
    X_test_in, y_test_in = X[div:], y[div:]

    testdata = testdata[:,:CL+1,:] # only use 10 timesteps for transformer predictions.rest of data is for icl experiments
    X_test_out, y_test_out = testdata[:,:-1,:], testdata[:,1:,:]

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_in_dataset = TensorDataset(X_test_in, y_test_in)
    test_out_dataset = TensorDataset(X_test_out, y_test_out)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_in_loader = DataLoader(test_in_dataset, batch_size=batch_size, shuffle=False)
    test_out_loader = DataLoader(test_out_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = Transformer(config).to(device)
    if loadmodel:
        print(f'Loading model from {loadmodel}')
        model.load_state_dict(torch.load(loadmodel, map_location=device))
        for name, param in model.named_parameters():
            param.requires_grad = False 

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
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            if 'linreg' in fname:
                loss = criterion(output[:, 0::2], batch_y[:,0::2])  # we only want the y values from model output.
            else:
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
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                total_test_in_loss += loss.item()
            for batch_X, batch_y in test_out_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
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

        if epoch % 500 == 0:
            torch.save(model.state_dict(), f'{totalbase}_model_epoch{epoch}.pth')

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

def train_many(Ls, Ws,title, traindata, testdata, CL, my_task_id, num_tasks):
    if my_task_id is None:
        my_task_id = int(sys.argv[1])
    if num_tasks is None:
        num_tasks = int(sys.argv[2])
    fnames = [(L,W) for L in Ls for W in Ws]
    my_fnames = fnames[my_task_id:len(fnames):num_tasks]
    print(my_fnames)
    for L,W in my_fnames:
        config = linreg_config()#get_default_config()
        config.n_layer = L
        config.n_embd = W
        config.max_seq_length = CL + 1
        train(config, traindata, testdata, fname = title, CL = CL)

if __name__ == '__main__':
    #generate_springdata(num_samples = 1000, sequence_length=50, plot = False)
    # datadict = torch.load('data/spring_data.pth')
    # traindata = datadict['sequences_train']
    # trainomegas = datadict['train_omegas']
    # testdata = datadict['sequences_test']
    # testomegas = datadict['test_omegas']
    #generate_dampedspringdata(num_samples = 10000, sequence_length=65, plot = True)
    # datadict = torch.load('data/dampedspring_data.pth')
    # datatype = 'underdamped'
    # traindata1 = datadict[f'sequences_train_{datatype}']
    # traindata1 = traindata1[torch.randperm(traindata1.size()[0])]
    # testdata1 = datadict[f'sequences_test_{datatype}']
    # testdata1 = testdata1[torch.randperm(testdata1.size()[0])]
    datadict = torch.load('data/linreg1_data.pth')
    traindata = datadict['traindata']
    testdata = datadict['testdata']
    CL = 2*65
    Ls = [1,2,3]
    Ws = [2,4,8,16]
    my_task_id = None
    num_tasks = None
    title = 'linreg1'
    train_many(Ls, Ws, title, traindata, testdata, CL, my_task_id, num_tasks)
    # traindata2 = datadict['sequences_train_overdamped']
    # traindata2 = traindata2[torch.randperm(traindata2.size()[0])]
    # testdata2 = datadict['sequences_test_overdamped']
    # testdata2 = testdata2[torch.randperm(testdata2.size()[0])]
    # trained_model2 = train(traindata2,testdata2, fname = 'springoverdamped', CL = CL)

    # traindata3 = torch.cat((datadict['sequences_train_underdamped'], datadict['sequences_train_overdamped']), dim = 0)
    # traindata3 = traindata3[torch.randperm(traindata3.size()[0])]
    # testdata3 = torch.cat((datadict['sequences_test_underdamped'], datadict['sequences_test_overdamped']), dim = 0)
    # testdata3 = testdata3[torch.randperm(testdata3.size()[0])]
    # trained_model3 = train(traindata3,testdata3, fname = 'springdamped', CL = CL)