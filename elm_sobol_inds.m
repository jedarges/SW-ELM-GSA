% elm_sobol_inds function
% by John Darges, 2023
%% Purpose:
% Computes the first-order and total Sobol' indices of a single layer neural network with phi(x) = e^x activation function
% Indices computed analytically from the network parameters
% NOTE: This code makes full use of MATLAB vectorization to avoid for-loops

%% Inputs: 
% 1. W - hidden layer weight matrix
% 2. beta - output weight vector
% 3. bias - bias vector

%% Outputs:
% 1. sobolR - vector of first-order Sobol' indices
% 2. sobolT - vector of total Sobol' indices
% 3. sig2 - variance of neural network

function [sobolR,sobolT,sig2] = elm_sobol_inds(W,beta,bias)

bias = reshape(bias,1,'');
beta = reshape(beta,'',1); N = length(beta);
W = reshape(W,'',N); ndim = size(W,1);

%% Surrogate of ELM
mu = sum(beta' .* exp(bias) .* prod(epsilon_eval(W),1)); 

%% Variance of ELM
beta_sum = beta * beta'; exp_bias_sum = exp(bias + bias');
cons = beta_sum .* exp_bias_sum;

epW = epsilon_eval(W);

E_plus = zeros(ndim,N,N); E_prod = zeros(ndim,N,N);
for j = 1:N
      E_plus(:,:,j) = epsilon_eval(W + W(:,j));
      E_prod(:,:,j) = diag(epW(:,j)) * epW;
end
E = reshape(prod(E_plus,1),N,N);
sig2 = sum(cons.*E','all') - mu^2;

%% Compute Sobol' indices
sobolR = zeros(ndim,1); sobolT = zeros(ndim,1);
for k = 1:ndim
    E_fo = E_prod; E_fo(k,:,:) = E_plus(k,:,:);
    S_k = reshape(prod(E_fo,1),N,N).*cons;
    sobolR(k) = 1/sig2 * (sum(S_k,'all') - mu^2);

    E_tot = E_plus; E_tot(k,:,:) = E_prod(k,:,:);
    S_Tk = reshape(prod(E_tot,1),N,N).*cons;
    sobolT(k) = 1 - 1/sig2 * (sum(S_Tk,'all') - mu^2);
end

    
    
%% Utility function    
function z = epsilon_eval(t)
    y = exp(t);
    z = (y - 1) ./ t;
   
    z(isnan(z))=1;
end
end
