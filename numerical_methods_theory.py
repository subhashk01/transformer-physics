import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from util import load_model, get_data, get_log_log_linear, linear_multistep_coefficients
import matplotlib.pyplot as plt
from math import factorial

'''
This file is used to test the various numerical methods we hypothesize 
the transformer could be using. We don't test the numerical methods on the Transformer.
Rather, we test it on ground truth data to get a stronger understanding of what
sort of errors we should experience and perhaps how to test better

The four methods we will test are:
- Runge-Kutta Non Linear Single Step (which is ultimately a form of the Taylor expansion for linear ODE)
- General Explicit Linear Multi Step Method
- General Implicit Linear Multi Step Method
- Exact formulation (only available for datasets where we know the analytical form)
'''


def get_A(gammas, omegas, X):
    # returns matrix A of size X.shape[0], X.shape[1], 2,2
    # where Ydot = Ay (spring equation, y = [x,v]
    matrix = torch.zeros((X.shape[0], X.shape[1], 2,2)) # 2x2 matrix for 2nd degree ODE
    matrix[:,:,0,0] = 0
    matrix[:,:,0,1] = 1
    matrix[:,:,1,0] = (-omegas**2).tile(X.shape[1],1).T
    matrix[:,:,1,1] = (-2*gammas).tile(X.shape[1],1).T
    return matrix

def plot_numerical_method(preds, ys, deltat, title):
    # plots MSE(preds[i], ys[i]) against deltat 
    targets = ['x', 'v']
    colors = 'bgrycmkw'
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (10,7))

    for order in range(len(preds)):
        pred = preds[order]
        y = ys[order]
        for i,target in enumerate(targets):
            MSE = ((pred[:,:,i] - y[:,:,i])**2).mean(dim=1)
            slope, intercept, r_value = get_log_log_linear(deltat, MSE)
            color = colors[order % len(colors)]
            mse = MSE.mean()
            axs[i].scatter(deltat, MSE, alpha = 1, color = color, label = f'{target}: Order {order+1}, MSE~{mse:.2e}\nlog(MSE) = {slope:.2f}log($\Delta t$)+{intercept:.2f}, R^2 = {r_value**2:.2f}')
    for i in range(len(targets)):
        axs[i].set_xlabel('deltat')
        axs[i].set_ylabel('MSE')
        axs[i].set_yscale('log')
        axs[i].set_xscale('log')
        axs[i].legend()

    fig.suptitle(title)
    plt.show()

def rk(maxterm = 5, datatype = 'damped', traintest = 'train', plot = True):
    gammas, omegas, sequences, times, deltat = get_data(datatype = datatype, traintest = traintest)
    terms = {}
    X,y = sequences[:,:-1, :], sequences[:,1:, :]
    matrix = get_A(gammas, omegas, X)
    terms['sequences'] = sequences

    deltat_tile = deltat.tile(X.shape[1], 1).T
    targets = ['x','v']
    coefs = []
    rk_terms = []
    
    for term in range(0, maxterm + 1):
        rk_term = torch.zeros(y.shape)
        coefs.append(1/factorial(term))
        pow = torch.linalg.matrix_power(matrix, term)
        ans = torch.zeros(pow.shape)
        for row in range(2): # row
            for col in range(2): # col
                ans[:,:,row,col] = pow[:,:,row,col] * X[:,:,col] * deltat_tile ** term
                terms[f'rk_{targets[row]}{term}{targets[col]}'] = ans[:, :, row, col]
        rk_term[:,:,0] = (terms[f'rk_x{term}x']+terms[f'rk_x{term}v'])*coefs[term]
        rk_term[:,:,1] = (terms[f'rk_v{term}x']+terms[f'rk_v{term}v'])*coefs[term]
        rk_terms.append(rk_term)
    rk_terms = torch.stack(rk_terms)
    preds = []
    ys = []
    MSEs = []
    for order in range(1,rk_terms.shape[0]):
        pred = rk_terms[:order+1].sum(dim=0)
        preds.append(pred)
        ys.append(y)
        mse = ((pred - y)**2).mean(dim=1)
        MSEs.append(mse)

    title = f'Runge Kutta Methods Efficacy\nOn {datatype} {traintest} Data'
    if plot:
        plot_numerical_method(preds, ys, deltat, title)
    torch.save(terms,f'data/underdampedspring_{datatype}_{traintest}_RK{maxterm}.pth')
    return terms, MSEs, preds, ys

def general_linear_multistep(maxsteps = 5, datatype = 'damped', traintest = 'train', plot = True):

    gammas, omegas, sequences, times, deltat = get_data(datatype = datatype, traintest = traintest)
    X,y = sequences[:,:-1,:], sequences[:,1:, :]
    terms = {}
    matrix = get_A(gammas, omegas, X)
    terms['sequences'] = sequences

    targets = ['x','v']
    y_backsteps = [torch.zeros(y.shape)] # implict step. not used for y
    f_backsteps = []
    f_backstep = torch.zeros(matrix.shape)  
    for row in range(len(targets)): # implicit step
        for col in range(len(targets)):
            f_backstep[:,:,row,col] = matrix[:,:,row,col] * y[:,:,col] * deltat.tile(X.shape[1],1).T
            terms[f'LMf_{targets[row]}0{targets[col]}'] = f_backstep[:,:,row,col]

    f_backsteps.append(f_backstep.sum(dim=3)) # implict step (backstep = 0)

    for backstep in range(1,maxsteps+1):
        y_backstep = torch.zeros(X.shape)
        f_backstep = torch.zeros(matrix.shape)
        for n in range(X.shape[1]):
            if n >= backstep - 1:
                for row in range(2):
                    for col in range(2):
                        f_backstep[:,n,row,col] = deltat * matrix[:,n,row,col] * X[:,n-backstep+1,col] # +1 bc we predicting y
                y_backstep[:,n] = X[:,n-backstep+1]
            
        for row in range(2):
            terms[f'LMy_{targets[row]}{backstep}'] = y_backstep[:,:,row]
            for col in range(2):
                terms[f'LMf_{targets[row]}{backstep}{targets[col]}'] = f_backstep[:,:,row,col]
        print(backstep)
        print(f_backstep.shape)
        print(f_backstep.sum(dim=3).shape)

        y_backsteps.append(y_backstep)
        f_backsteps.append(f_backstep.sum(dim = 3))
    torch.save(terms,f'data/underdampedspring_{datatype}_{traintest}_LM{maxsteps}.pth')
    f_backsteps = torch.stack(f_backsteps)
    y_backsteps = torch.stack(y_backsteps)
    exp_preds = []
    ys_exp = []
    imp_preds = []
    ys_imp = []
    MSE_exp = []
    MSE_imp = []
    for order in range(1,maxsteps+1):
        exp_pred = get_linear_multistep_pred(order, f_backsteps, y_backsteps, explicit = True)[:,order-1:]
        exp_preds.append(exp_pred)
        y_exp = y[:,order-1:]
        ys_exp.append(y_exp)
        MSE_exp.append(((y_exp-exp_pred)**2).mean(dim=1))

        imp_pred = get_linear_multistep_pred(order, f_backsteps, y_backsteps, explicit = False)[:, max(0,order-2):]
        imp_preds.append(imp_pred)
        y_imp = y[:, max(0,order-2):]
        ys_imp.append(y_imp)
        MSE_imp.append(((y_imp-imp_pred)**2).mean(dim=1))
    exp_title = f'Linear Multistep Explicit Method (Adams Bashforth)\nOn {datatype} {traintest} Data'
    imp_title =  f'Linear Multistep Implicit Method (Adams Moulton)\nOn {datatype} {traintest} Data'
    if plot:
        plot_numerical_method(exp_preds, ys_exp, deltat, exp_title)
        plot_numerical_method(imp_preds, ys_imp, deltat, imp_title)

    preds = {'exp_preds': exp_preds, 'ys_exp': ys_exp, 'imp_preds':imp_preds, 'ys_imp': ys_imp}
    return terms, MSE_exp, MSE_imp, preds

def get_linear_multistep_pred(order, f_backsteps, y_backsteps, explicit = True):
    if explicit:
        f = f_backsteps[1:]
        y = y_backsteps[1:]
        # adams bashworth method coefficeints
    else:
        f = f_backsteps
        y = y_backsteps
        # adams moulton method coefficients
    all_coefficients = linear_multistep_coefficients(explicit)
    coefficients = all_coefficients[order]
    f_sum = torch.zeros(f[0].shape)
    y_sum = torch.zeros(y[0].shape)

    for i, ycoef in enumerate(coefficients[0]):
        y_sum += ycoef * y[i]
    for i, fcoef in enumerate(coefficients[1]):
        f_sum += fcoef * f[i]
    pred = f_sum+y_sum
    return pred


def test_ICL_LM(order = 10,  method = 'LM_imp', datatype = 'underdamped', traintest = 'train'):

    if method[:2] == 'LM':
        _, _, _, preds = general_linear_multistep(maxsteps = order, datatype = datatype, traintest = traintest, plot=False)
        ei = method[-3:]
        ys = preds[f'ys_{ei}'] # expects imp or exp
        ypreds = preds[f'{ei}_preds']
        title = f'Linear Multistep {ei} ICL Test'
    elif method[:2] == 'rk':
        _, _, ypreds, ys = rk(maxterm = order, datatype = datatype, traintest = traintest, plot = False)
        title = 'Runge Kutta ICL Test'
    criterion = nn.MSELoss()
    plt.figure(figsize = (10,10))
    for o in range(order):
        y, ypred = ys[o], ypreds[o]
        mses = []
        CLs = []
        for CL in range(1,y.shape[1]):
            mse = criterion(y[:,:CL], ypred[:,:CL])
            mses.append(mse)
            CLs.append(CL+o)
        slope, intercept, r_value = get_log_log_linear(CLs, mses)
        plt.plot(CLs, mses, label = rf'Order {o+1}, log(mse) = {slope:.2f}log(CLs)+{intercept:.2f}, $R^2$ = {r_value**2:.2f}')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Context Length')
    plt.title(title)
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
        




def compare_methods(highest_order = 5, datatype = 'damped', traintest = 'train'):
    _, MSE_imp, MSE_exp, _ = general_linear_multistep(highest_order, datatype, traintest, False)
    _, MSE_rk = rk(highest_order, datatype, traintest, False)
    MSE_imp = torch.stack(MSE_imp)
    MSE_exp = torch.stack(MSE_exp)
    MSE_rk = torch.stack(MSE_rk)
    colors = 'rgb'
    fig, axs = plt.subplots(nrows = MSE_rk.shape[0], ncols = MSE_rk.shape[-1])
    for i,target in enumerate(['x', 'v']):
        for order in range(5):#MSE_rk.shape[0]):
            ax = axs[order,i]
            mserk, mseexp, mseimp = MSE_rk[order,:,i], MSE_exp[order,:,i], MSE_imp[order,:, i]
            ax.hist(mserk, alpha = 0.1, color = colors[0], label = f'RK-{order}')
            ax.hist(mseexp, alpha = 0.1, color = colors[1], label = f'AB-{order}')
            ax.hist(mseimp, alpha = 0.1, color = colors[2], label = f'AM-{order}')
            ax.set_xscale('log')
            ax.set_yscale('log')
            print(f'{target}, ORDER {order+1}')
            logrk, logexp, logimp = torch.log10(mserk), torch.log10(mseexp), torch.log10(mseimp)
            
            print(f'RK: {logrk.mean():.2f} mean, {logrk.std():.2f} std')
            print(f'AB: {logexp.mean():.2f} mean, {logexp.std():.2f} std')
            print(f'AM: {logimp.mean():.2f} mean, {logimp.std():.2f} std')
        print()
            
            
def exp_weights(datatype = 'underdamped', traintest = 'train'):
    
    gamma, omega0, data, times, deltat = get_data(datatype = datatype, traintest = traintest)
    X,y = data[:,:-1,:], data[:,1:,:]
    omegas = torch.sqrt(omega0**2 - gamma**2)
    prefactor = torch.exp(-gamma*deltat)
    beta = - (gamma**2 + omegas**2)/omegas
    cos = torch.cos(omegas*deltat)
    sin = torch.sin(omegas*deltat)
    w00 = (cos + gamma/omegas*sin) * prefactor
    w01 = (sin/omegas) * prefactor
    w10 = (beta * sin) * prefactor
    w11 = (cos - gamma/omegas*sin) * prefactor
    w00 = w00.unsqueeze(1)
    w01 = w01.unsqueeze(1)
    w10 = w10.unsqueeze(1)
    w11 = w11.unsqueeze(1)


    x = X[:,:,0]
    v = X[:,:,1]

    matrixtarget = {
                    'w00': w00 * x,
                    'w01': w01 * v,
                    'w10': w10 * x,
                    'w11': w11 * v,
                    'sequences': data
        }
    torch.save(matrixtarget, f'data/{datatype}spring_{datatype}_{traintest}_mw.pth')


def matrix_weights_prediction(datatype = 'underdamped', traintest = 'train'):
    gamma, omega0, data, times, deltat = get_data(datatype = datatype, traintest = traintest)
    X, y = data[:,:-1,:], data[:,1:,:]
    matrixtarget = torch.load(f'data/{datatype}spring_{datatype}_{traintest}_mw.pth')
    xpred = matrixtarget['w00'] + matrixtarget['w01']
    vpred = matrixtarget['w10'] + matrixtarget['w11']
    pred = torch.stack([xpred, vpred], dim = 2)
    MSE = ((pred - y)**2).mean()
    print(MSE)



if __name__ == '__main__':
    exp_weights()
    order = 10
    #test_ICL_LM(explicit = True)
    # rk(order, datatype = 'underdamped',traintest = 'train', plot = False)
    #general_linear_multistep(order, datatype = 'underdamped',traintest = 'train', plot = False)
    #lm = torch.load('data/underdampedspring_underdamped_train_LM10.pth')
    #print(lm['LMf_x4x'])
    # methods = ['LM_exp', 'rk']
    # for method in methods:
    #     test_ICL_LM(method = method)
    #compare_methods()
    #print(ans)
    #exp_weights()
    matrix_weights_prediction()
    # get_data()