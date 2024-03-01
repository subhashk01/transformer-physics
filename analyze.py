import torch
import matplotlib.pyplot as plt
from model import Transformer
from config import get_default_config
import numpy as np
import torch.nn as nn

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

def load_model(file='models/spring_16emb_2layer_10CL_20000epochs_0.001lr_64batch_model.pth'):
    config = get_default_config()
    model = Transformer(config)
    model_state = torch.load(file)
    model.load_state_dict(model_state)
    model.eval()
    return model

def loss_deltat_omega_relationship(model):
    datadict = torch.load('data/spring_data.pth')
    traindata = datadict['sequences_train']
    trainomegas = datadict['train_omegas']
    testdata = datadict['sequences_test']
    testomegas = datadict['test_omegas']
    traintimes = datadict['train_times']
    testtimes = datadict['test_times']
    train_deltat = traintimes[:,1] - traintimes[:,0]
    test_deltat = testtimes[:,1] - testtimes[:,0]

    model.eval()
    Xtrain,ytrain = traindata[:,:-1,:],traindata[:,1:,:]
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

        plt.show()

def plot_model_predictions(model):
    datadict = torch.load('data/spring_data.pth')
    
    n = 4
    #CL = 2**n
    CL = 4
    #indices = np.linspace(start = 0, stop = traindata.shape[1]-1, num = CL+1, dtype = int)
    traindata = datadict['sequences_train'][:,:CL+1,:]
    trainomegas = datadict['train_omegas']
    testdata = datadict['sequences_test'][:,:CL+1,:]
    testomegas = datadict['test_omegas']

    traintimes = datadict['train_times']
    testtimes = datadict['test_times']
    deltat_train = traintimes[:,1] - traintimes[:,0]
    deltat_test = testtimes[:,1] - testtimes[:,0]

    


    # randomize all test data
    # randomize test data
    indices = np.random.permutation(testdata.shape[0])
    testdata = testdata[indices]
    testomegas = testomegas[indices]
    deltat_test = deltat_test[indices]
    


    model.eval()
    Xtrain,ytrain = traindata[:,:-1,:],traindata[:,1:,:]
    Xtest,ytest = testdata[:,:-1,:],testdata[:,1:,:]

    


    with torch.no_grad():
        ytrain_pred = model(Xtrain)
        MSEs_train = (ytrain_pred - ytrain)**2
        ytest_pred = model(Xtest)
        MSEs_test = (ytest_pred - ytest)**2
        for i in range(3):
            plt.figure(figsize = (10,6))

            plt.plot(ytrain[i,:,0], ytrain[i,:,1].numpy(), 'b-', label = 'Training y Data')
            plt.plot(ytrain_pred[i,:,0], ytrain_pred[i,:,1].numpy(), 'r--', label = 'Training y Predictions')

            plt.scatter(ytrain[i,0,0], ytrain[i,0,1].numpy(), c = 'b', label = 'Data Start')
            plt.scatter(ytrain_pred[i,0,0], ytrain_pred[i,0,1].numpy(), c = 'r', label = 'Pred Start')

            plt.scatter(ytrain[i,-1,0], ytrain[i,-1,1].numpy(), marker = '*',s = 100, c = 'b', label = 'Data End')
            plt.scatter(ytrain_pred[i,-1,0], ytrain_pred[i,-1,1].numpy(), s = 100, marker = '*', c = 'r', label = 'Pred End')
            plt.xlabel('x = cos(wt)')
            plt.ylabel('v = -wsin(wt)')
            plt.title(f'Training Data and Prediction, CL = {CL} \nDatum {i}: Omega = {trainomegas[i].item():.2f}, deltaT = {deltat_train[i].item():.2f}, MSE = {MSEs_train[i].mean().item():.2e}')
            plt.legend()
            plt.show()
        for j in range(0):
            # do the same thing but for test data
            plt.figure(figsize = (10,6))
            plt.plot(ytest[j,:,0], ytest[j,:,1].numpy(), 'b-', label = 'Test y Data')
            plt.plot(ytest_pred[j,:,0], ytest_pred[j,:,1].numpy(), 'r--', label = 'Test y Predictions')

            plt.scatter(ytest[j,0,0], ytest[j,0,1].numpy(), c = 'b', label = 'Data Start')
            plt.scatter(ytest_pred[j,0,0], ytest_pred[j,0,1].numpy(), c = 'r', label = 'Pred Start')

            plt.scatter(ytest[j,-1,0], ytest[j,-1,1].numpy(), marker = '*',s = 100, c = 'b', label = 'Data End')
            plt.scatter(ytest_pred[j,-1,0], ytest_pred[j,-1,1].numpy(), s = 100, marker = '*', c = 'r', label = 'Pred End')
            plt.xlabel('x = cos(wt)')
            plt.ylabel('v = -wsin(wt)')
            plt.title(f'Training Data and Prediction, CL = {CL} \nDatum {j}: Omega = {testomegas[j].item():.2f}, deltaT = {deltat_test[j].item():.2f}, MSE = {MSEs_test[j].mean().item():.2e}')
            plt.legend()
            plt.show()
        

def study_ICL(model, plotex = False):
    model.eval()
    datadict = torch.load('data/spring_data.pth')
    traindata =  datadict['sequences_train']
    div = int(0.8*traindata.shape[0])
    testindata = datadict['sequences_train'][div:]
    testoutdata = datadict['sequences_test']

    context_length = []
    MSE_ins = []
    MSE_outs = []

    floor_val = min(traindata.shape[1], config.max_seq_length)
    floor_val = int(np.floor(np.log2(floor_val)))
    print(floor_val)
    testindata = testindata[:,:2**floor_val+1,:]
    testoutdata = testoutdata[:,:2**floor_val+1,:]
    print(testindata.shape)
    
    for n in range(floor_val+1):
        with torch.no_grad():
            CL = 2**n
            context_length.append(CL)
            indices = np.linspace(start = 0, stop = testindata.shape[1]-1, num = CL+1)
            testin = testindata[:,:CL+1,:] #indices.astype(int)
            testout = testoutdata[:,:CL+1,:]
            #how tf ru making predictions about one data point?

            Xin, yin = testin[:,:-1,:], testin[:,1:,:]
            breakpoint()
            Xout, yout = testout[:,:-1,:], testout[:,1:,:]
            ypred_in = model(Xin)
            ypred_out = model(Xout)


            criterion = torch.nn.MSELoss()
            MSE_in = criterion(ypred_in, yin).item()
            print(CL, MSE_in)

            deltayin = ypred_in - yin
            print(deltayin[:5])
            print()

            MSE_ins.append(MSE_in)
            MSE_out = criterion(ypred_out, yout).item()
            MSE_outs.append(MSE_out)


            if plotex:
                for ind in range(3):
                    plt.figure(figsize = (10,6))
                    plt.title(f'Context Length = {CL}, Datum {ind}')    
                    print(yin.shape, 'yo')
                    yin = yin.squeeze()
                    ypred_in = ypred_in.squeeze()
                    plt.plot(testin[ind, :, 0].numpy(), testin[ind, :, 1].numpy(), 'b--', label = 'Input Data')
                    plt.scatter(yin[ind, 0].item(), yin[ind, 1].item(), c = 'b', label = 'Input Data End')
                    plt.scatter(ypred_in[ind, 0].item(), ypred_in[ind, 1].item(), c = 'b',marker = '*', label = 'Input Prediction End')
                    plt.legend()
                    plt.show()
            
    plt.figure(figsize = (10,6))
    plt.axvline(x = 10, color = 'k', linestyle = '--', label = 'Training Data Length')
            
            
    plt.plot(context_length, MSE_ins, color = 'b',marker = 'o', label = 'Test-In MSE')
    plt.plot(context_length, MSE_outs, color = 'r', marker = 'o', label = 'Test-Out MSE')
    #make tick marks powers of 2
    # Set the x-axis to logarithmic scale
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)

    # Manually set the ticks and labels after setting the logarithmic scale
    logs = [int(np.log2(n)) for n in context_length]
    plt.xticks(ticks=context_length, labels=[rf'$2^{{{log}}}$' for log in logs])

    plt.xlabel('Context Length')
    plt.ylabel('MSE Over Dataset')
    plt.title('MSE vs Context Length with In/Out of Distribution Test Data')
    plt.legend()
    plt.show()

            


if __name__ == '__main__':
    model = load_model()
    # new_max_seq_length = 1024  # New desired maximum sequence length
    # new_positional_embeddings = nn.Parameter(torch.zeros(new_max_seq_length, model.n_embed))
    # new_positional_embeddings[:model.max_seq_length].data = model.positional_embeddings.data

    # model.positional_embeddings = new_positional_embeddings
    # model.max_seq_length = new_max_seq_length



    #plot_loss_curves()
    #loss_deltat_omega_relationship(model)
    #plot_model_predictions(model)
    study_ICL(model, plotex = True)
    #print('here')
    #plot_model_predictions(model)
