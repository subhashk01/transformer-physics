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
        

        torch.save(best_probe, f'probes/eulerlinear/eulerlinear_{target}target_{modeltype}_layer{layer}_neuron{neuron}_Linear_probe.pth')
        best_weights = probe.l.weight.squeeze()
        for i, targetname in enumerate(targetnames):
            print(f'{targetname}: {best_weights[i].item():.2f}')
    return best_MSE, best_R2, best_weights, MSE_probe_actual_best, MSE_model_actual

def predict_allneurons_euler(layer = 2, CL = 65, target='x', modeltype='underdamped', epochs=100000, lr=1e-3):
    neurons = list(range(CL))
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    model.eval()

    data = torch.load('data/dampedspring_data.pth')
    data = torch.cat((data[f'sequences_test_damped'], data[f'sequences_train_damped']))
    # we want to take only the first CL+1 timesteps
    # randomize first dimension of data
    indices = torch.randperm(data.shape[0])
    data = data[indices]
    data = data[:5000,:CL+1,:]
    X = data[:,:-1,:]
    _, targetnames = eulermapping_data(0, target)

    _, hidden_states = model.forward_hs(X)
    hidden_states = torch.stack(hidden_states)
    hidden_states = hidden_states.transpose(0, 1)

    all_inputs = []
    y_targets = []
    ind = {'x': 0, 'v': 1}
    for neuron in neurons:
        inputs = []
        for targetname in targetnames:
            hs = hidden_states[:, layer, neuron, :]
            probe = LinearProbe(hs.shape[1])
            probepath = f'probes/{modeltype}/{targetname}_layer{layer}_neuron{neuron}_Linear_probe.pth'
            probe.load_state_dict(torch.load(probepath))
            input = probe(hs).squeeze()
            inputs.append(input)
        inputs = torch.stack(inputs, dim = 1)
        all_inputs.append(inputs)
        y_targets.append(data[:,neuron+1, ind[target]])
    all_inputs = torch.stack(all_inputs)
    y_targets = torch.stack(y_targets)
    
    # collaps first two dims of all_inputs into one
    all_inputs = all_inputs.view(all_inputs.shape[0]*all_inputs.shape[1], all_inputs.shape[2])
    y_targets = y_targets.view(y_targets.shape[0]*y_targets.shape[1])
    # Randomize the order of the inputs
    indices = torch.randperm(all_inputs.shape[0])
    all_inputs = all_inputs[indices]
    y_targets = y_targets[indices]
    div = int(0.8 * all_inputs.shape[0])
    Xtrain = all_inputs[:div].detach()
    ytrain = y_targets[:div].detach()
    Xtest = all_inputs[div:].detach()
    ytest = y_targets[div:].detach()

    probe = LinearDirect(Xtrain.shape[1])
    epoch_pbar = tqdm(range(epochs), desc='Training Probe')

    optimizer = optim.Adam(probe.parameters(), lr=lr)
    lambda_l1 = 1e-7  # Regularization strength
    criterion = nn.MSELoss()

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
        epoch_pbar.set_postfix({'Train Loss': loss.item(), 'Test Loss': test_loss.item(),'R^2 Test': r2_test})

    torch.save(probe.state_dict(), f'probes/eulerlinear/eulerlinear_{target}target_{modeltype}_layer{layer}_allneurons_Linear_probe.pth')
    for i, targetname in enumerate(targetnames):
        print(f'{targetname}: {probe.l.weight.squeeze()[i].item():.2f}')


def compare_euler_model(run = True, modeltype = 'underdamped', CL = 65):
    # THEORY: if euler was well approximating the model and not just data,
    # we expect MSE(model, euler) < MSE(euler, data), MSE(model, data)
    neurons = list(range(CL))
    model = load_model(f'models/spring{modeltype}_16emb_2layer_65CL_20000epochs_0.001lr_64batch_model.pth')
    model.eval()

    data = torch.load('data/dampedspring_data.pth')
    data = torch.cat((data[f'sequences_test_damped'], data[f'sequences_train_damped']))
    # we want to take only the first CL+1 timesteps
    data = data[:100,:CL+1,:]
    X, y = data[:,:-1,:], data[:,1:,:]
    y_model = model(X)

    ind = {'x': 0, 'v': 1}
    criterion = nn.MSELoss()

    _, hidden_states = model.forward_hs(X)
    hidden_states = torch.stack(hidden_states)
    hidden_states = hidden_states.transpose(0, 1)

    layer = 2
    for target in ['x', 'v']:
        MSEs_probe_model = []
        MSEs_probe_actual = []
        MSEs_model_actual = []
        eulerprobe_outputs = []
        for neuron in neurons:
            print(target, neuron)
            if run:
                MSE_probe_model, r2, _, MSE_probe_actual, MSE_model_actual = predict_neuron_euler(target = target, neuron = neuron, epochs = 10000)
            else:
                hs = hidden_states[:, layer, neuron, :]
                _, targetnames = eulermapping_data(neuron = neuron, target = target)
                inputs = []
                for targetname in targetnames:
                    probe = LinearProbe(hs.shape[1])
                    probepath = f'probes/{modeltype}/{targetname}_layer{layer}_neuron{neuron}_Linear_probe.pth'
                    probe.load_state_dict(torch.load(probepath))
                    input = probe(hs).squeeze()
                    inputs.append(input)
                inputs = torch.stack(inputs, dim = 1).detach()

                eulerprobe = LinearDirect(inputs.shape[1])
                #eulerprobepath = f'probes/eulerlinear/eulerlinear_{target}target_{modeltype}_layer{layer}_neuron{neuron}_Linear_probe.pth'
                eulerprobepath = f'probes/eulerlinear/eulerlinear_{target}target_{modeltype}_layer{layer}_allneurons_Linear_probe.pth'
                eulerprobe.load_state_dict(torch.load(eulerprobepath))
                eulerprobe_output = eulerprobe(inputs).squeeze().detach()
                eulerprobe_outputs.append(eulerprobe_output)
                eulerprobe_stack = torch.stack(eulerprobe_outputs).transpose(0,1).detach()

                # reverse dims of eulerprobe


                
                y_model_neuron = y_model[:, :neuron+1, ind[target]].detach()
                y_neuron = y[:, :neuron+1, ind[target]].detach()

                MSE_model_actual = criterion(y_model_neuron, y_neuron)
                MSE_probe_model = criterion(eulerprobe_stack, y_model_neuron)
                MSE_probe_actual = criterion(eulerprobe_stack, y_neuron)
                MSEs_model_actual.append(MSE_model_actual)
                MSEs_probe_model.append(MSE_probe_model)
                MSEs_probe_actual.append(MSE_probe_actual)

        CLs = [neuron+1 for neuron in neurons]
        plt.plot(CLs, MSEs_probe_model, label = 'MSE(euler probe, transformer)', color = 'r', marker = 'o')
        plt.plot(CLs, MSEs_probe_actual, label = 'MSE(euler probe, actual data)', color = 'b', marker = 'o')
        plt.plot(CLs, MSEs_model_actual, label = 'MSE(transformer, actual data)', color = 'g', marker = 'o')
        
        plt.legend()
        plt.title(f'MSE of {modeltype} Models Predicting {target}')
        plt.xlabel('CL')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('MSE')
        plt.show()


def plot_R2euler_datatypes(targetname='x0x', layer=2, neuron=0, modeltype='underdamped', CL=65):
    # plots how well the euler method approximates underdamped/overdamped data with the underdamped model
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

key = 'underdamped'
        data = torch.load('data/dampedspring_data.pth')
        deltat = data[f'times_test_{key}'][:, 1] - data[f'times_test_{key}'][:, 0]
        deltat =torch.cat((deltat, data[f'times_train_{key}'][:, 1] - data[f'times_train_{key}'][:, 0]), dim=0).unsqueeze(1)
        omega0= torch.cat((data[f'omegas_test_{key}'], data[f'omegas_train_{key}']), dim=0).unsqueeze(1)
        gamma= torch.cat((data[f'gammas_test_{key}'], data[f'gammas_train_{key}']), dim=0).unsqueeze(1)
        omegas = torch.sqrt(omega0**2 - gamma**2)
        data = torch.cat((data[f'sequences_test_{key}'], data[f'sequences_train_{key}']))
        x = data[:,:,0]
        v = data[:,:,1]

        prefactor = torch.exp(-gamma*deltat)
        beta = - (gamma**2 + omegas**2)/omegas
        cos = torch.cos(omegas*deltat)
        sin = torch.sin(omegas*deltat)
        w00 = (cos + gamma/omegas*sin) * prefactor
        w01 = (sin/omegas) * prefactor
        w10 = (beta * sin) * prefactor
        w11 = (cos - gamma/omegas*sin) * prefactor
        # prefactor = torch.exp(-gamma*deltat)
        # beta = - (gammas**2 + omegas**2)/omegas
        # cos = torch.cos(omegas*deltat)
        # sin = torch.sin(omegas*deltat)
        # w00 = cos + gammas/omegas*sin
        # w01 = sin/omegas
        # w10 = beta * sin
        # w11 = cos - gammas/omegas*sin
        # terms = [w00, w01, w10, w11]

        #breakpoint()


        targetdict = {
                    # 'omega0':omega0, 
                    # 'omega0^2': omega0**2, 
                    # 'omega\lambda': omega, 
                    # 'omega^2\lambda^2':omega**2, 
                    # 'gammadivomega': gamma/omega,
                    # 'vprec_under': (gamma**2+omega**2)/omega, 
                    # 'gammas':gamma,
                    # 'deltat':deltat,
                    # 'omega0^2deltat': omega0**2*deltat,
                    # 'gammadeltat': gamma*deltat,
                    # 'omega0deltat': omega0*deltat,
                    # 'omega0^2deltat^2': omega0**2*deltat**2,
                    # 'gammadeltat^2': gamma*deltat**2,
                    # 'gammaomega0^2deltat^2': gamma*omega0**2*deltat**2,
                    # 'fourgamma^2minusomega0^2deltat^2': (4*gamma**2-omega0**2)*deltat**2,
                    # 'x0': x0,
                    # 'v0deltat': v0*deltat,
                    # 'v0': v0,
                    # 'omega0^2deltatx0': omega0**2*deltat*x0,
                    # 'gammadeltatv0': gamma*deltat*v0,
                    'x0x': x,
                    'x1v': deltat * v,
                    'x2v': deltat**2 * gamma * v,
                    'x2x': deltat**2 * omega0**2 * x,
                    'x3v': deltat**3 * (4 * gamma**2 - omega0**2) * v,
                    'x3x': deltat**3 * gamma * omega0**2 * x,
                    'x4v': deltat**4 * (-2 * gamma**3 + omega0**2 * gamma) * v,
                    'x4x': deltat**4 * (-4 * gamma**2 * omega0**2 + omega0**4) * x,

                    'v0v': v,
                    'v1v': deltat * gamma * v,
                    'v1x': deltat * omega0**2 * x,
                    'v2v': deltat**2 * (4 * gamma**2 - omega0**2) * v,
                    'v2x': deltat**2 * gamma * omega0**2 * x,
                    'v3v': deltat**3 * (-2 * gamma**3 + omega0**2 * gamma) * v,
                    'v3x': deltat**3 * (-4 * gamma**2 * omega0**2 + omega0**4) * x,
                    'v4v': deltat**4 * (16 * gamma**4 - 12 * omega0**2 * gamma**2 + omega0**4) * v,
                    'v4x': deltat**4 * (2 * gamma**3 * omega0**2 - omega0**4 * gamma) * x

                    }

        matrixtarget = {
                    'w00': w00 * x,
                    'w01': w01 * v,
                    'w10': w10 * x,
                    'w11': w11 * v
        }

        basicdata = {
            'sequences': data,
            'omegas': omega0,
            'gammas': gamma,
            'deltat': deltat
        }


        # save targetdict
        torch.save({**targetdict, **basicdata}, f'data/euler_terms_{modelkey}.pth')
        torch.save({**matrixtarget, **basicdata}, f'data/matrix_terms_{modelkey}.pth')
        