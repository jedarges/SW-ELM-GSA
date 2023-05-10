% Function elm_train_model
% by John Darges, 2023
%% 
%% Purpose: 
% This code takes training data and creates ELM weights and biases according to user specifications. 
% Solves the linear least squares problem by solving the L2 regularized normal equations using the MATLAB backslash command

%% Inputs
%  1. Xd - Array of training data inputs: (training size) x (input dimension) array
%  2. Yd - Vector of model output at training points: (training size) x 1 vector
%  3. model_data - Object which contains user specifications: sparisification parameter and regularization parameter

%% Outputs
% 1. W - Hidden layer weight matrix: (input dimension) x (number of neurons/basis functions) array
% 2. bias - Bias vector: 1 x (number of neurons/basis functions) vector
% 3. beta - Output weight vector: (number of neurons/basis functions) x 1 vector

function [W,bias,beta] = elm_train_model(Xd,Yd,model_data)
Yd = reshape(Yd,'',1); s_sz = length(Yd);
Xd = reshape(Xd,s_sz,''); ndim = size(Xd,2);

%% User choice parameters
p = model_data.p;
lambda = model_data.lambda;
N = model_data.nneurons;

%% Activation function
phi = @(x) exp(x);

%% Create sparse weight matrix
W = randn(ndim,N);
Z = rand(ndim,N) > p;
W = W .* Z;
% Bias vector
bias = randn(1,N);

%% Solve least squares
H = phi(Xd * W + bias);
G = H'*H; G_dim = length(G);
beta = (G + (lambda * eye(G_dim))) \ (H'*Yd);

end
