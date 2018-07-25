rng(1)
num_samples=60;
num_simulations = 100;
num_repetitions=100;

%this vector will include the proportion of junk simulation for which the
%complex model was preferred over the junk model.
p_junkapproach = nan(num_repetitions,1);

%this vector will include 1 if the complex model was preferred, 0
%otherwise.
p_bic = nan(num_repetitions,1);

%a repetition is a new dataset. Like running a new experiment, not like
%generating new random data.
for repetition_i=1:num_repetitions
    
    if mod(repetition_i,100)==0
        repetition_i
    end
    
    x = zscore(rand(num_samples,1));
    noise = zscore(rand(size(x)));
    
    %you can play with the coefficients here. For the values below, the
    %complex model should be preferred.
    y = x + x.^2+noise;

    % polynomial (over-complicated) model
    x_powers = [x.^2, x.^3, x.^4];
    [~,~,r_complex] = regress(y,[x,x_powers]);
    %equation from:
    %https://www.statlect.com/fundamentals-of-statistics/linear-regression-maximum-likelihood
    loglikelihood_complex = -(num_samples/2)*(log(2*pi)-log(var(x)))-(1/(2*var(x)))*var(r_complex);
    bic_complex = -2*loglikelihood_complex+log(num_samples)*6;
    
    [~,~,r_simple] = regress(y,x);
    loglikelihood_simple = -(num_samples/2)*(log(2*pi)-log(var(x)))-(1/(2*var(x)))*var(r_simple);
    bic_simple = -2*loglikelihood_simple+log(num_samples)*3;

    loglikelihood_junk_model = zeros(num_simulations,1);
    for simulation_i = 1:num_simulations
        junk_regressors = zscore(rand(num_samples,3));
        [~,~,r_junk] = regress(y,[x junk_regressors]);
        loglikelihood_junk_model(simulation_i) = ...
             -(num_samples/2)*(log(2*pi)-log(var(x)))-(1/(2*var(x)))*var(r_junk);
    end
    
    prop_complex_model_was_better = sum(loglikelihood_junk_model<=loglikelihood_complex)/num_simulations;
    p_junkapproach(repetition_i) = prop_complex_model_was_better;
    p_bic(repetition_i) = bic_simple>bic_complex;
end
