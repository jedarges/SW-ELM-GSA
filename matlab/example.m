% This code demonstrates how to implement SW-ELM to compute the first-order and total Sobol' indices of an example model.
%       There are three sections:
%       1. "Example setup" creates the example model we are looking at.
%       2. "ELM Setup" defines the parameters for building the ELM. How many basis functions to use? What regularziation parameter to use? Which sparsification parameters to test?
%       3. "Generate training and validation sets" creates data sets from the example model
%       4. "SW-ELM" is where we actually build the surrogate and performs GSA
%       5. "Plotting" plots the results


%% 1. Example setup
example_fcn = @(x) sum(sin(x)); % example is additive sum of sine functions
% x(1) ~ U([-1,1]) % x(2) ~ U([1,2]) % x(3) ~ U([0,5]) x(i) ~ U([0,1]), i > 3 % this defines what uniform distribution each input belongs to
ndim = 10; % what is the input dimension?
trials = 1; % do we want to test multiple realizations of the weights?


%% ELM Settings
model_data.nneurons = 500; % Number of hidden layer neurons, i.e., ELM basis functions
model_data.lambda = 1e-4; % L2 regularization parameter, you should choose this using your favorite method, i.e. L-curve, cross validation, etc.
p_list = [0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.99]; % Here you list all the sparsification parameters to test. More parameters means more linear solves.
p_sz = length(p_list); % Number of sparsification parameters to test

%% Generate training and validation sets
s_sz = 1000; % Training set size
Xd_train = lhsdesign(s_sz,ndim); Yd_train = zeros(s_sz,1); % lhsdesign samples are from [0,1]^ndim
for i = 1:s_sz
    Xi = Xd_train(i,:); Xi(1) = 2 * Xi(1) - 1; Xi(2) = Xi(2) + 1; Xi(3) = 5 * Xi(3); % linearly transform 
    Yd_train(i) = example_fcn(Xi); 
end

va_sz = 300; % Validation set size
Xd_valid = lhsdesign(s_sz,ndim); Yd_valid = zeros(s_sz,1);
for k = 1:s_sz
    Xk = Xd_valid(k,:); Xk(1) = 2 * Xk(1) - 1; Xk(2) = Xk(2) + 1; Xk(3) = 5 * Xk(3); 
    Yd_valid(k) = example_fcn(Xk);
end

%% SW-ELM
rel = zeros(p_sz,1);
for j = 1:p_sz
    model_data.p = p_list(j);
    for l = 1:trials
        [W,bias,beta] = elm_train_model(Xd_train,Yd_train,model_data); % 
        elm_valid = exp(Xd_valid * W + bias) * beta;
        rel(j) = rel(j) + norm(Yd_valid - elm_valid) / norm(Yd_valid); % relative error estimate that we use to choose which sparsification parameter is best
    end
end

p_opt = p_list(rel == min(rel)); 
model_data.p = p_opt;
[W,bias,beta] = elm_train_model(Xd_train,Yd_train,model_data);
elm_valid = exp(Xd_valid * W + bias) * beta; 
error = norm(Yd_valid - elm_valid) / norm(Yd_valid); % error of SW-ELM

[sobolR,sobolT,sig2] = elm_sobol_inds(W,beta,bias); % finally this function computes the sensitivity indices


%% Plotting
figure;
plot(p_list,rel,'.-','MarkerSize',25)
set(gca,'FontSize',15)
xlabel('P(W_{i,j}=0)')
ylabel('Relative Error (L2)')


figure;
bar([sobolR sobolT])
set(gca,'FontSize',15)
ylabel('Sobol'' Index')
legend('First-order','Total')
