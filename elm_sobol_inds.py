import numpy as np

def epsilon_eval(t):
    y = np.exp(t)
    z = (y - 1) / t
    z[np.isnan(z)] = 1
    return z

def elm_sobol_inds(W, beta, bias):
    bias = bias.reshape(1,-1)
    beta = beta.reshape(-1,1)
    N = len(beta)
    W = W.reshape(-1,N)
    ndim = W.shape[0]

    ## Surrogate of ELM
    mu = np.sum(beta.T * np.exp(bias) * np.prod(epsilon_eval(W),axis=0))

    ## Variance of ELM
    beta_sum = beta.dot(beta.T)
    exp_bias_sum = np.exp(bias + bias.T)
    cons = beta_sum * exp_bias_sum

    epW = epsilon_eval(W)
    E_plus = np.zeros((ndim,N,N))
    E_prod = np.zeros((ndim,N,N))
    for j in range(N):
        E_plus[:,:,j] = epsilon_eval(W + W[:,j,None])
        E_prod[:,:,j] = np.diag(epW[:,j]) @ epW

    E = np.prod(E_plus,axis=0).reshape(N,N)
    sig2 = np.sum(cons*E) - mu**2

    ## Compute Sobol' indices
    sobolR = np.zeros(ndim)
    sobolT = np.zeros(ndim)
    for k in range(ndim):
        E_fo = E_prod.copy()
        E_fo[k,:,:] = E_plus[k,:,:]
        S_k = np.prod(E_fo,axis=0).reshape(N,N) * cons
        sobolR[k] = 1/sig2 * (np.sum(S_k) - mu**2)

        E_tot = E_plus.copy()
        E_tot[k,:,:] = E_prod[k,:,:]
        S_Tk = np.prod(E_tot,axis=0).reshape(N,N) * cons
        sobolT[k] = 1 - 1/sig2 * (np.sum(S_Tk) - mu**2)

    return sobolR, sobolT, sig2
