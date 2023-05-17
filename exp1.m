clc
clearvars
%addpath(genpath("/Users/sloan/MATLAB/YALMIP-master"))
%addpath(genpath("/Users/sloan/MATLAB/sedumi-master"))
%addpath(genpath("/Library/gurobi1001/macos_universal2/matlab/"))

d = 10; % dimension
sample_sizes = [10, 20, 50, 75, 100];%, 200, 500];
T = 20; % iterations
rho = 0.1; % Wp perturbation size (here, will be a translation)
eps = 0.05; % TV perturbation size
sigma = 2; % moment bound, specific to distributions here
contamination_scale_factor = 10; % push to make standard WDRO have unbounded excess risk

dual_norm = 2;
verbose = 0; % you can set it to 1 if you want to see the solver's progress

theta = randn(d,1); % coefficients for affine hypothesis
theta = theta./sqrt(theta'*theta); % normalize to unit vector
theta_0 = randn(1); % bias for affine hypothesis

standard_excess_risks = zeros(T,length(sample_sizes));
outlier_robust_true_mean_excess_risks = zeros(T,length(sample_sizes));
outlier_robust_excess_risks = zeros(T,length(sample_sizes));

for t = 1:T
    disp("iteration:")
    disp(t)
    n = sample_sizes(end);
    X = randn(n,d); % true features, each row is one sample
    theta = randn(d,1); % coefficients for affine hypothesis
    theta = theta./sqrt(theta'*theta); % normalize to unit vector
    theta_0 = randn(1); % bias for affine hypothesis
    noise = 0.01 * randn(n, 1);
    y = X * theta + theta_0 + noise; % true labels under affine hypothesis plus noise
    
    theta_tilde = contamination_scale_factor * -theta; % large coefficient vector opposite to theta
    translation = zeros(1,d);
    translation(1) = rho;
    X_tilde = X + repmat(translation,n,1); % translate all feature vectors by rho
    X_tilde(1:floor(eps*n),:) = X_tilde(1:floor(eps*n),:)*contamination_scale_factor;
    y_tilde = X_tilde * theta + theta_0 + noise;
    y_tilde(1:floor(eps*n)) = X_tilde(1:floor(eps*n),:) * theta_tilde + theta_0 + noise(1:floor(eps*n)); % use theta_tilde for eps fraction of labels
    
    reordering = randperm(n);
    X_tilde = X_tilde(reordering,:);
    y_tilde = y_tilde(reordering);

    % risk of true coefficients
    best_risk = mean(abs(X * theta + theta_0 - y));
    
    for i = 1:length(sample_sizes)
        m = sample_sizes(i);
        disp("sample size")
        disp(m)
        % using standard WDRO on corrupted data
        disp("standard DRO excess risk")
        [theta_star, theta_0_star, obj_star] = regular_WDRO(X_tilde(1:m,:), y_tilde(1:m), 0.001, dual_norm, verbose);
        standard_excess_risk = mean(abs(X * theta_star + theta_0_star - y)) - best_risk;
        disp(standard_excess_risk)
        standard_excess_risks(t,i) = standard_excess_risk;
        
        % using outlier-robust WDRO, proper sigma, true mean as center
        disp("outlier-robust DRO excess risk (true mean)")
        z_0 = zeros(11, 1);
        z_0(end) = theta_0;
        [theta_star2, theta_0_star2, obj_star2] = outlier_robust_WDRO(X_tilde(1:m,:), y_tilde(1:m), sigma, rho, 0, z_0, dual_norm, verbose);
        outlier_robust_true_mean_excess_risk = mean(abs(X * theta_star2 + theta_0_star2 - y)) - best_risk;
        disp(outlier_robust_true_mean_excess_risk)
        outlier_robust_true_mean_excess_risks(t,i) = outlier_robust_true_mean_excess_risk;

        % using outlier-robust WDRO, proper sigma, true mean as center
        disp("outlier-robust DRO excess risk (estimated mean)")
        z_0 = cheap_robust_mean_estimate(X_tilde, 2*eps);
        z_0(end+1) = cheap_robust_mean_estimate(y_tilde, 2*eps);
        z_0 = z_0';
        [theta_star2, theta_0_star2, obj_star2] = outlier_robust_WDRO(X_tilde(1:m,:), y_tilde(1:m), sigma, rho, 0, z_0, dual_norm, verbose);
        outlier_robust_excess_risk = mean(abs(X * theta_star2 + theta_0_star2 - y)) - best_risk;
        disp(outlier_robust_excess_risk)
        outlier_robust_excess_risks(t,i) = outlier_robust_excess_risk;
    end
end

%% save data
save(strcat("exp1_",datestr(now)),"standard_excess_risks", "outlier_robust_excess_risks", "outlier_robust_true_mean_excess_risks")

%% load data
load

%% bootstrapping


% d = 10; % dimension
% sample_sizes = [10, 20, 50, 75, 100];%, 200, 500];
% T = 20; % iterations
% rho = 0.1;

K = 100; % number of bootstrap resamples
standard_averages = mean(standard_excess_risks,1);
outlier_robust_true_mean_averages = mean(outlier_robust_true_mean_excess_risks,1);
outlier_robust_averages = mean(outlier_robust_excess_risks,1);

standard_bootstraps = zeros(K, length(sample_sizes));
outlier_robust_true_mean_bootstraps = zeros(K, length(sample_sizes));
outlier_robust_bootstraps = zeros(K, length(sample_sizes));
for k = 1:K
    for i = 1:length(sample_sizes)
        n = sample_sizes(i);
        standard_bootstraps(k,i) = mean(randsample(standard_excess_risks(:,i),T,true));
        outlier_robust_true_mean_bootstraps(k,i) = mean(randsample(outlier_robust_true_mean_excess_risks(:,i),T,true));
        outlier_robust_bootstraps(k,i) = mean(randsample(outlier_robust_excess_risks(:,i),T,true));
    end
end
standard_bootstraps = sort(standard_bootstraps,1);
outlier_robust_true_mean_bootstraps = sort(outlier_robust_true_mean_bootstraps,1);
outlier_robust_bootstraps = sort(outlier_robust_bootstraps,1);
standard_p10 = standard_bootstraps(round(K*.1),:);
standard_p90 = standard_bootstraps(round(K*.9),:);
outlier_robust_true_mean_p10 = outlier_robust_true_mean_bootstraps(round(K*.1),:);
outlier_robust_true_mean_p90 = outlier_robust_true_mean_bootstraps(round(K*.9),:);
outlier_robust_p10 = outlier_robust_bootstraps(round(K*.1),:);
outlier_robust_p90 = outlier_robust_bootstraps(round(K*.9),:);

%% plots
hold on
errorbar(sample_sizes, standard_averages + rho, standard_averages - standard_p10 + rho, standard_p90 - standard_averages + rho);
errorbar(sample_sizes, outlier_robust_true_mean_averages + rho, outlier_robust_true_mean_averages - outlier_robust_true_mean_p10 + rho, outlier_robust_true_mean_p90 - outlier_robust_true_mean_averages + rho);
errorbar(sample_sizes, outlier_robust_averages + rho, outlier_robust_averages - outlier_robust_p10 + rho, outlier_robust_p90 - outlier_robust_averages + rho);

title("Excess Risk for Varied Sample Size and Method")
xlabel("# samples")
%ylim([0,3])
xlim([10,100])
ylabel("excess risk (mean absolute deviation)")
legend("standard WDRO","outlier-robust WDRO w/ true mean", "outlier-robust WDRO w/ estimated mean")
hold off