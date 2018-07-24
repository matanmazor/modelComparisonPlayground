clear
clc

rng(1)
num_samples=60;
num_simulations = 1000;

% Real model: y ~ x + noise
x = zscore(rand(num_samples,1));
noise = zscore(rand(size(x)));
y = x + noise;

% polynomial (over-complicated) model
x_powers = [x.^2, x.^3, x.^4];
[~,~,r_complex] = regress(y,[x,x_powers]);
%equation from:
%https://www.statlect.com/fundamentals-of-statistics/linear-regression-maximum-likelihood
loglikelihood_complex = -(num_samples/2)*(log(2*pi)-log(var(x)))-1/((2*var(x))*var(r_complex));

[~,~,r_simple] = regress(y,x);
loglikelihood_simple = -(num_samples/2)*(log(2*pi)-log(var(x)))-1/((2*var(x))*var(r_simple));

% We eould expect a junk model would work just as well as the complex model
loglikelihood_junk_model = zeros(num_simulations,1);
h=waitbar(0);
for simulation_i = 1:num_simulations
    junk_regressors = zscore(rand(num_samples,3));
    [~,~,r_junk] = regress(y,[x junk_regressors]);
    loglikelihood_junk_model(simulation_i) = ...
         -(num_samples/2)*(log(2*pi)-log(var(x)))-1/((2*var(x))*var(r_junk));
    waitbar(simulation_i/num_simulations,h,'Running Simulations')
end
close(h)
figure;
histogram(loglikelihood_complex - loglikelihood_junk_model);
xlabel(sprintf(['log likelihood Complex model minus log_likelihood junk models\n'...
    '(Positive values suggest complex polynomial model was better than simple linear model)']))
prop_complex_model_was_better = sum(loglikelihood_junk_model<=loglikelihood_complex)/num_simulations;
fprintf('\nProportion of simulations real model was better = %.2f\n',prop_complex_model_was_better)
if prop_complex_model_was_better<=0.5
    fprintf('Suggesting the simple model was better\n')
else
    fprintf('Suggesting the complex model was better\n')
end
