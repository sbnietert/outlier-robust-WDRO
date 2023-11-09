%% setup

clc
clearvars
%addpath(genpath("/Users/???/MATLAB/YALMIP-master"))
%addpath(genpath("/Users/???/MATLAB/sedumi-master"))
%addpath(genpath("/Library/gurobi1001/macos_universal2/matlab/"))

d = 10; % feature dimension
k = 3; % label dimension
sample_sizes = [10, 20, 50, 75, 100];%, 200, 500];
T = 10; % iterations
rho = 0.1; % Wp perturbation size (here, will be a translation)
eps = 0.1; % TV perturbation size
sigma = sqrt(d + k); % moment bound (sqrt(dim) * covariance)
contamination_scale_factor = 10; % push to make standard WDRO have unbounded excess risk

dual_norm = 2;
verbose = 0; % set to 1 to see the solver's progress

%% compute excess risks

standard_excess_risks = zeros(T,length(sample_sizes));
outlier_robust_excess_risks = zeros(T,length(sample_sizes));

for t = 1:T
    disp("iteration:")
    disp(t)

    for i = 1:length(sample_sizes)
        n = sample_sizes(i);
        disp("sample size")
        disp(n)

        X = randn(n,d); % true features, each row is one sample
        theta_star = randn(10,k);
        y = X * theta_star;
    
        % risk of true coefficients
        best_risk = mean(sum(abs(y - X*theta_star),2)); % = 0
    
        X_tilde = X;
        y_tilde = y;
        % TV perturbation
        X_tilde(1:floor(eps*n),:) = X(1:floor(eps*n),:) * contamination_scale_factor;
        y_tilde(1:floor(eps*n),:) = -contamination_scale_factor^2 * y(1:floor(eps*n),:);
        % Wp perturbation
        translation = zeros(1,d);
        translation(1) = rho;
        X_tilde = X_tilde + repmat(translation,n,1); 
    
        % shuffle (shouldn't matter)
        reordering = randperm(n);
        X_tilde = X_tilde(reordering,:);
        y_tilde = y_tilde(reordering,:);

        % using standard WDRO on corrupted data
        disp("standard DRO excess risk")
        [theta_hat, ~] = regular_WDRO_k_dim(X_tilde, y_tilde, rho, dual_norm, verbose);
        standard_excess_risks(t,i) = mean(sum(abs(y - X*theta_hat),2)) - best_risk;
        disp(standard_excess_risks(t,i))

        z_0 = cheap_robust_mean_estimate(X_tilde, 2*eps);
        z_0(end+1:end+k) = cheap_robust_mean_estimate(y_tilde, 2*eps);

        % using outlier-robust WDRO
        disp("outlier-robust DRO excess risk")
        [theta_hat2, ~] = outlier_robust_WDRO_k_dim(X_tilde, y_tilde, sigma, rho, eps, z_0', dual_norm, verbose);
        outlier_robust_excess_risks(t,i) = mean(sum(abs(y - X*theta_hat2),2)) - best_risk;
        disp(outlier_robust_excess_risks(t,i))
   end
end

%% save data

save(strcat("exp4_k_dim_",datestr(now)),"standard_excess_risks", "outlier_robust_excess_risks")

%% bootstrapping

K = 100; % number of bootstrap resamples
standard_averages = mean(standard_excess_risks,1);
outlier_robust_averages = mean(outlier_robust_excess_risks,1);

standard_bootstraps = zeros(K, length(sample_sizes));
outlier_robust_bootstraps = zeros(K, length(sample_sizes));
for k = 1:K
    for i = 1:length(sample_sizes)
        n = sample_sizes(i);
        standard_bootstraps(k,i) = mean(randsample(standard_excess_risks(:,i),T,true));
        outlier_robust_bootstraps(k,i) = mean(randsample(outlier_robust_excess_risks(:,i),T,true));
    end
end
standard_bootstraps = sort(standard_bootstraps,1);
outlier_robust_bootstraps = sort(outlier_robust_bootstraps,1);
standard_p10 = standard_bootstraps(round(K*.1),:);
standard_p90 = standard_bootstraps(round(K*.9),:);
outlier_robust_p10 = outlier_robust_bootstraps(round(K*.1),:);
outlier_robust_p90 = outlier_robust_bootstraps(round(K*.9),:);

%% plots

hold on
errorbar(sample_sizes, standard_averages, standard_averages - standard_p10, standard_p90 - standard_averages);
errorbar(sample_sizes, outlier_robust_averages, outlier_robust_averages - outlier_robust_p10, outlier_robust_p90 - outlier_robust_averages);
set(gca, 'YScale', 'log')
title("\rm Excess Linear Regression Risk with WDRO")
ax = gca;
ax.TitleFontSizeMultiplier = 1.5;
xl = xlabel("# samples");
xlim([10,100]);
fontsize(xl,"increase")
fontsize(xl,"increase")
yl = ylabel("excess risk (ℓ₁ mean absolute deviation)");
fontsize(yl,"increase")
fontsize(yl,"increase")
leg = legend("standard WDRO","OR-WDRO");
fontsize(leg,"increase")
fontsize(leg,"increase")
hold off