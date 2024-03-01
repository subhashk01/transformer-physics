import torch
from util import set_seed
import matplotlib.pyplot as plt

set_seed(0)

def generate_linregdata(num_samples = 1000, sequence_length=10):
    #NOTE: not built to handle multidimensional x data. have to change delimiter stuff

    # Ensure that |w| > 0.1 and w is in range [-1, 1]
    w = torch.rand(num_samples) * 0.9 + 0.1  # Random weights in range [0.1, 1]
    w = w * torch.where(torch.rand(num_samples) > 0.5, torch.tensor(1.0), torch.tensor(-1.0))  # Randomly flip sign

    # Ensure that |x1| > 0.1 and x1 is in range [-1, 1]
    x1 = torch.rand(num_samples) * 0.9 + 0.1  # Random initial x values in range [0.1, 1]
    x1 = x1 * torch.where(torch.rand(num_samples) > 0.5, torch.tensor(1.0), torch.tensor(-1.0))  # Randomly flip sign

    delimiter = (torch.rand(num_samples) * (1-x1)/9.).unsqueeze(1)  # Random delimiters in range [0, (1-x1)/9]

    indices = torch.arange(sequence_length+1).unsqueeze(0).repeat(num_samples, 1)  # Indices for each position in the sequence
    x_values = x1.unsqueeze(1) + indices * delimiter  # Broadcasted addition to generate x values
    y_values = w.unsqueeze(1) * x_values  # Element-wise multiplication for y = wx

    sequences = torch.stack((x_values, y_values), dim=2)  # Shape: (num_samples, sequence_length, 2)
    print(sequences.size())
    return sequences

def generate_springdata(num_samples = 1000, sequence_length=10, plot = False):
    # Generate x = cos(wt), v = -w*sin(wt). trains on omegas between 0.5pi and 1pi, tests on 0.25pi-0.5pi and 1pi-1.25pi
    omegas_range = [0.25*torch.pi, 1.25*torch.pi]
    delta_omega  = omegas_range[1]-omegas_range[0]
    
    train_omegas = torch.rand(num_samples) * delta_omega/2 + omegas_range[0] + delta_omega/4
    # middle half of the omega interval is the training set
    train_deltat = torch.rand(num_samples) * 2*torch.pi / (train_omegas)

    start = 1
    skip = 1
    train_times = torch.arange(start, start+skip*sequence_length+1, step = skip).unsqueeze(0).repeat(num_samples, 1)
    train_times = train_times * train_deltat.unsqueeze(1)

    x_train = torch.cos(train_omegas.unsqueeze(1) * train_times)
    v_train = -train_omegas.unsqueeze(1) * torch.sin(train_omegas.unsqueeze(1) * train_times)
    # stack x and v
    sequences_train = torch.stack((x_train, v_train), dim=2)  # Shape: (num_samples, sequence_length, 2)
    
    test_omegas_low = torch.rand(num_samples//4) * delta_omega/4 + omegas_range[0]
    test_omegas_high = torch.rand(num_samples//4) * delta_omega/4 + omegas_range[1] - delta_omega/4
    #concatenate the two
    test_omegas = torch.cat((test_omegas_low, test_omegas_high))
    test_deltat = torch.rand(test_omegas.shape[-1]) * 2*torch.pi / test_omegas

    test_times = torch.arange(start, start+skip*sequence_length+1, step = skip).unsqueeze(0).repeat(test_deltat.shape[-1], 1)
    test_times = test_times * test_deltat.unsqueeze(1)

    x_test = torch.cos(test_omegas.unsqueeze(1) * test_times)
    v_test = -test_omegas.unsqueeze(1) * torch.sin(test_omegas.unsqueeze(1) * test_times)
    # stack x and v
    sequences_test = torch.stack((x_test, v_test), dim=2)  # Shape: (num_samples, sequence_length, 2)

    # Save data
    torch.save({
        'sequences_train': sequences_train,
        'train_omegas': train_omegas,
        'sequences_test': sequences_test,
        'test_omegas': test_omegas,
        'train_times': train_times,
        'test_times': test_times
    }, 'data/spring_data.pth')
    if plot:
        plt.hist(train_omegas, color = 'b', label = 'Training Omegas', bins=20)
        plt.hist(test_omegas, color = 'r', label = 'Test Omegas', bins=20)
        plt.xlabel('Omega')
        plt.ylabel('Frequency')
        plt.title(f'{train_omegas.shape[0]} Training and {test_omegas.shape[0]} Test Omegas')
        plt.legend()
        plt.show()
    return sequences_train, train_omegas, sequences_test, test_omegas

def plot_training_data():
    datadict = torch.load('data/spring_data.pth')
    traindata = datadict['sequences_train']
    trainomegas = datadict['train_omegas']
    testdata = datadict['sequences_test']
    testomegas = datadict['test_omegas']
    traintimes = datadict['train_times']
    testtimes = datadict['test_times']
    for ind in range(5):
        fig, axs = plt.subplots(1,2, figsize = (20,6))
        ax = axs[0]
        ax.set_title('First 10 data points')
        t = traindata[:,1:11,:].numpy()
        print(t.shape)
        ax.plot(t[ind,:,0], t[ind,:,1], 'b--', label = 'Training y Data, Omega = {trainomegas[ind].item():.2f}')

        ax = axs[1]
        ax.set_title('Last 10 data points')
        t = traindata[:,-10:, :].numpy()
        ax.plot(t[ind,:,0], t[ind,:,1], 'r--', label = 'Training y Data, Omega = {trainomegas[ind].item():.2f}')


        plt.xlabel('x')
        plt.ylabel('v')
        plt.title(f'Training/Test Data Index {ind}. Annotated w/ Timestep (not given to model)')
        plt.legend()
        plt.show()


    

if __name__ == '__main__':
    generate_springdata(num_samples = 1000, sequence_length=1025, plot = False)
    #plot_training_data()
