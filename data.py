import torch
from util import set_seed
import matplotlib.pyplot as plt

set_seed(10)

def generate_springdata(num_samples = 1000, sequence_length=10, plot = False):
    # Generate x = cos(wt), v = -w*sin(wt). trains on omegas between 0.5pi and 1pi, tests on 0.25pi-0.5pi and 1pi-1.25pi
    omegas_range = [0.25*torch.pi, 1.25*torch.pi]
    delta_omega  = omegas_range[1]-omegas_range[0]
    
    train_omegas = torch.rand(num_samples) * delta_omega/2 + omegas_range[0] + delta_omega/4
    # middle half of the omega interval is the training set
    train_deltat = torch.rand(num_samples) * 2*torch.pi / (train_omegas) # cos(wt) has period 2pi/w. so deltat>2pi/w is redundant

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
    }, f'data/spring_data.pth')
    if plot:
        plt.hist(train_omegas, color = 'b', label = 'Training Omegas', bins=20)
        plt.hist(test_omegas, color = 'r', label = 'Test Omegas', bins=20)
        plt.xlabel('Omega')
        plt.ylabel('Frequency')
        plt.title(f'{train_omegas.shape[0]} Training and {test_omegas.shape[0]} Test Omegas')
        plt.legend()
        plt.show()
    return sequences_train, train_omegas, sequences_test, test_omegas


def omega1to2(w1, w2, num_samples = 1000, sequence_length=10):
    # creates data with deltats input sequences are w1 and output sequences are w2
    #w2 should be less than w1
    assert(w2 <= w1)
    w1s = torch.ones(num_samples) * w1
    train_times = torch.arange(1, sequence_length+2).unsqueeze(0).repeat(num_samples, 1)
    train_deltat = torch.rand(num_samples) * 2*torch.pi / (w1s) # cos(wt) has period 2pi/w. so deltat>2pi/w is redundant
    train_times = train_times * train_deltat.unsqueeze(1)

    x_train_w1 = torch.cos(w1s.unsqueeze(1) * train_times)
    v_train_w1 = -w1s.unsqueeze(1) * torch.sin(w1s.unsqueeze(1) * train_times)
    sequences_w1 = torch.stack((x_train_w1, v_train_w1), dim=2)  # Shape: (num_samples, sequence_length, 2)

    w2s = torch.ones(num_samples) * w2
    x_train_w2 = torch.cos(w2s.unsqueeze(1) * train_times)
    v_train_w2 = -w2s.unsqueeze(1) * torch.sin(w2s.unsqueeze(1) * train_times)
    sequences_w2 = torch.stack((x_train_w2, v_train_w2), dim=2)  # Shape: (num_samples, sequence_length, 2)

    sequences_w1 = sequences_w1[:,:-1,:]
    sequences_w2 = sequences_w2[:,1:,:]
    return sequences_w1, sequences_w2


def generate_dampedspringdata(num_samples = 1000, sequence_length=10, plot = False):
    # Generate x = cos(wt), v = -w*sin(wt). trains on omegas between 0.5pi and 1pi, tests on 0.25pi-0.5pi and 1pi-1.25pi
    omegas_range = [0.5*torch.pi, 2.5*torch.pi]
    delta_omega  = omegas_range[1]-omegas_range[0]
    
    train_omegas = torch.rand(num_samples) * delta_omega/2 + omegas_range[0] + delta_omega/4
    # middle half of the omega interval is the training set
    train_deltat = torch.rand(num_samples) * 2*torch.pi / (sequence_length * train_omegas) # cos(wt) has period 2pi/w. so deltat>2pi/w is redundant

    start = 1
    skip = 1
    train_times = torch.arange(start, start+skip*sequence_length+1, step = skip).unsqueeze(0).repeat(num_samples, 1)
    train_times = train_times * train_deltat.unsqueeze(1)

    num_under_train = int(0.5 * train_times.shape[0])


    test_omegas_low = torch.rand(num_samples//4) * delta_omega/4 + omegas_range[0]
    test_omegas_high = torch.rand(num_samples//4) * delta_omega/4 + omegas_range[1] - delta_omega/4


    #concatenate the two
    test_omegas = torch.cat((test_omegas_low, test_omegas_high))
    # randomize test_omegas
    test_omegas = test_omegas[torch.randperm(test_omegas.shape[0])]

    test_deltat = torch.rand(test_omegas.shape[-1]) * 2*torch.pi / (sequence_length * test_omegas)

    test_times = torch.arange(start, start+skip*sequence_length+1, step = skip).unsqueeze(0).repeat(test_deltat.shape[-1], 1)
    test_times = test_times * test_deltat.unsqueeze(1)

    num_under_test = int(0.5 * test_times.shape[0])


    def gen_underdamped(omegas_0, gammas, train_times):
        omegas = torch.sqrt(omegas_0**2 - gammas**2)
        wt = omegas.unsqueeze(1) * train_times
        gt = gammas.unsqueeze(1) * train_times
        gw = gammas.unsqueeze(1) / omegas.unsqueeze(1)
        vcoeff = (gammas.unsqueeze(1)**2+omegas.unsqueeze(1)**2) / (omegas.unsqueeze(1))
        x = torch.exp(-gt) * (torch.cos(wt) + gw * torch.sin(wt))
        v = -torch.exp(-gt) * vcoeff * torch.sin(wt)
        seq = torch.stack((x, v), dim=2)
        return seq

    # UNDERDAMPED
    omegas_train_under = train_omegas[:num_under_train]
    gammas_train_under = torch.rand(num_under_train) * omegas_train_under
    times_train_under = train_times[:num_under_train]
    sequences_train_under = gen_underdamped(omegas_train_under, gammas_train_under, times_train_under)

    omegas_test_under = test_omegas[:num_under_test]
    gammas_test_under = torch.rand(num_under_test) * omegas_test_under
    times_test_under = test_times[:num_under_test]
    sequences_test_under = gen_underdamped(omegas_test_under, gammas_test_under, times_test_under)

    # if plot:
    #     plt.figure(figsize=(10,10))
    #     for i in range(4):
    #         plt.subplot(2,2,i+1)
    #         i  = omegas_test_under.shape[0]-i-1
    #         plt.plot(times_train_under[i],sequences_train_under[i,:,0], label = 'x')
    #         plt.plot(times_train_under[i],sequences_train_under[i,:,1], label = 'v')
    #         plt.title(f'Omega: {omegas_train_under[i]:.2f}, Gamma: {gammas_train_under[i]:.2f}')
    #         plt.legend()
    #     plt.show()

     


    # plot sample curves
    

    def gen_overdamped(omegas_0, gammas, train_times):
        lam = torch.sqrt(gammas**2 - omegas_0**2)
        lamt = lam.unsqueeze(1) * train_times
        glam = gammas.unsqueeze(1)/lam.unsqueeze(1)
        gt = gammas.unsqueeze(1) * train_times
        exp_prec = torch.exp(-gt)/2
        x = exp_prec * ((1+glam)*torch.exp(lamt) + (1-glam)*torch.exp(-lamt))
        v = exp_prec * ((1+glam)*(lam.unsqueeze(1)- gammas.unsqueeze(1))*torch.exp(lamt) - (1-glam)*(lam.unsqueeze(1)+gammas.unsqueeze(1))*torch.exp(-lamt))
        seq = torch.stack((x, v), dim=2)
        return seq

    omegas_train_over = train_omegas[num_under_train:]
    omega_max = omegas_train_over.max()+delta_omega/4
    gammas_train_over = torch.rand(omegas_train_over.shape[0]) * (omega_max-omegas_train_over)+omegas_train_over
    times_train_over = train_times[num_under_train:]
    sequences_train_over = gen_overdamped(omegas_train_over, gammas_train_over, times_train_over)

    omegas_test_over = test_omegas[num_under_test:]
    omega_max = omegas_test_over.max()+delta_omega/4
    gammas_test_over = torch.rand(omegas_test_over.shape[0]) * (omega_max-omegas_test_over)+omegas_test_over
    times_test_over = test_times[num_under_test:]
    sequences_test_over = gen_overdamped(omegas_test_over, gammas_test_over, times_test_over)
    # if plot:
    #     plt.figure(figsize=(10,10))
    #     for i in range(4):
    #         plt.subplot(2,2,i+1)
    #         i  = omegas_test_under.shape[0]-i-1
    #         plt.plot(times_test_over[i],sequences_test_over[i,:,0], label = 'x')
    #         plt.plot(times_test_over[i],sequences_test_over[i,:,1], label = 'v')
    #         plt.title(f'Omega: {omegas_test_over[i]:.2f}, Gamma: {gammas_test_over[i]:.2f}')
    #         plt.legend()
    #     plt.show()
 
    sequences_train_damped = torch.cat((sequences_train_under, sequences_train_over, ), dim = 0)
    sequences_test_damped = torch.cat((sequences_test_under, sequences_test_over), dim = 0)
    omegas_train_damped = torch.cat((omegas_train_under, omegas_train_over), dim = 0)
    omegas_test_damped = torch.cat((omegas_test_under, omegas_test_over), dim = 0)
    times_train_damped = torch.cat((times_train_under, times_train_over), dim = 0)
    times_test_damped = torch.cat((times_test_under, times_test_over), dim = 0)
    gammas_train_damped = torch.cat((gammas_train_under, gammas_train_over), dim = 0)
    gammas_test_damped = torch.cat((gammas_test_under, gammas_test_over), dim = 0)



    torch.save({
        'sequences_train_underdamped': sequences_train_under,
        'omegas_train_underdamped': omegas_train_under,
        'times_train_underdamped': times_train_under,
        'gammas_train_underdamped': gammas_train_under,

        'sequences_test_underdamped': sequences_test_under,
        'omegas_test_underdamped': omegas_test_under,
        'times_test_underdamped': times_test_under,
        'gammas_test_underdamped': gammas_test_under,

        'sequences_train_overdamped': sequences_train_over,
        'omegas_train_overdamped': omegas_train_over,
        'times_train_overdamped': times_train_over,
        'gammas_train_overdamped': gammas_train_over,
        
        'sequences_test_overdamped': sequences_test_over,
        'omegas_test_overdamped': omegas_test_over,
        'times_test_overdamped': times_test_over,
        'gammas_test_overdamped': gammas_test_over,

        'sequences_train_damped': sequences_train_damped,
        'omegas_train_damped': omegas_train_damped,
        'times_train_damped': times_train_damped,
        'gammas_train_damped': gammas_train_damped,
        
        'sequences_test_damped': sequences_test_damped,
        'omegas_test_damped': omegas_test_damped,
        'times_test_damped': times_test_damped,
        'gammas_test_damped': gammas_test_damped,




    }, f'data/dampedspring_data.pth')
    # if plot:
    #     plt.hist(train_omegas, color = 'b', label = 'Training Omegas', bins=20)
    #     plt.hist(test_omegas, color = 'r', label = 'Test Omegas', bins=20)
    #     plt.xlabel('Omega')
    #     plt.ylabel('Frequency')
    #     plt.title(f'{train_omegas.shape[0]} Training and {test_omegas.shape[0]} Test Omegas')
    #     plt.legend()
    #     plt.show()
    #return sequences_train, train_omegas, sequences_test, test_omegas

if __name__ == '__main__':
    generate_dampedspringdata(num_samples = 10000, sequence_length=65, plot = False)
    #plot_training_data()
    #playground()
