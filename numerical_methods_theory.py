import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from util import load_model, get_data
import matplotlib.pyplot as plt
from math import factorial
from scipy.stats import linregress
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
            xlog = np.log(deltat)
            ylog = np.log(MSE)
            slope, intercept, r_value, p_value, std_err = linregress(xlog, ylog)
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
    return terms, MSEs

def general_linear_multistep(maxsteps = 5, datatype = 'damped', traintest = 'train', plot = True):

    gammas, omegas, sequences, times, deltat = get_data(datatype = datatype, traintest = traintest)
    X,y = sequences[:,:-1,:], sequences[:,1:, :]
    terms = {}
    matrix = get_A(gammas, omegas, X)

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

    
    return terms, MSE_exp, MSE_imp

def get_linear_multistep_pred(order, f_backsteps, y_backsteps, explicit = True):
    if explicit:
        f = f_backsteps[1:]
        y = y_backsteps[1:]
        # adams bashworth method coefficeints
        all_coefficients = {
                1: [[1], [1]],
                2: [[1], [3/2, -1/2]],
                3: [[1], [23/12, -16/12, 5/12]],
                4: [[1], [55/24, -59/24, 37/24, -9/24]],
                5: [[1], [1901/720, -2774/720, 2616/720, -1274/720, 251/720]],
                6: [[1], [4277/1440, -7923/1440, 9982/1440, -7298/1440, 2877/1440, -475/1440]],
                7: [[1], [198721/60480, -447288/60480, 705549/60480, -688256/60480, 407139/60480, -134472/60480, 19087/60480]],
                8: [[1], [434241/120960, -1152169/120960, 2183877/120960, -2664477/120960, 2102243/120960, -1041723/120960, 295767/120960, -36799/120960]],
                9: [[1], [14097247/3628800, -43125206/3628800, 95476786/3628800, -139855262/3628800, 137968480/3628800, -91172642/3628800, 38833486/3628800, -9664106/3628800, 1070017/3628800]],
                10: [[1], [30277247/7257600, -104995189/7257600, 265932123/7257600, -454661776/7257600, 538363838/7257600, -444772162/7257600, 252618224/7257600, -94307320/7257600, 20884811/7257600, -2082753/7257600]]
            } # coefs given as y, f
    else:
        f = f_backsteps
        y = y_backsteps
        # adams moulton method coefficients
        all_coefficients = {
                1: [[0, 1], [1]],
                2: [[0, 1], [1/2, 1/2]],
                3: [[0, 1], [5/12, 2/3, -1/12]],
                4: [[0, 1], [9/24, 19/24, -5/24, 1/24]],
                5: [[0, 1], [251/720, 646/720, -264/720, 106/720, -19/720]],
                6: [[0, 1], [475/1440, 1427/1440, -798/1440, 482/1440, -173/1440, 27/1440]],
                7: [[0, 1], [19087/60480, 65112/60480, -46461/60480, 37504/60480, -20211/60480, 6312/60480, -863/60480]],
                8: [[0, 1], [36799/120960, 139849/120960, -121797/120960, 123133/120960, -88547/120960, 41499/120960, -11351/120960, 1375/120960]],
                9: [[0, 1], [1070017/3628800, 4467094/3628800, -4604594/3628800, 5595358/3628800, -5033120/3628800, 3146338/3628800, -1291214/3628800, 312874/3628800, -33953/3628800]],
                10: [[0, 1], [2082753/7257600, 9449717/7257600, -11271304/7257600, 16002320/7257600, -17283646/7257600, 13510082/7257600, -7394032/7257600, 2687864/7257600, -583435/7257600, 57281/7257600]]
            }

    coefficients = all_coefficients[order]
    f_sum = torch.zeros(f[0].shape)
    y_sum = torch.zeros(y[0].shape)

    for i, ycoef in enumerate(coefficients[0]):
        y_sum += ycoef * y[i]
    for i, fcoef in enumerate(coefficients[1]):
        f_sum += fcoef * f[i]
    pred = f_sum+y_sum
    return pred


def compare_methods(highest_order = 5, datatype = 'damped', traintest = 'train'):
    _, MSE_imp, MSE_exp = general_linear_multistep(highest_order, datatype, traintest, False)
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
            
            
    #plt.show()

if __name__ == '__main__':
    order = 10
    rk(order, datatype = 'underdamped',traintest = 'train', plot = False)
    general_linear_multistep(order, datatype = 'underdamped',traintest = 'train', plot = False)
    #compare_methods()
    #print(ans)
    # get_data()