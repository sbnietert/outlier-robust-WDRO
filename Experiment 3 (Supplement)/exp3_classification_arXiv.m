%% setup

clc
clearvars
%addpath(genpath("/Users/???/MATLAB/YALMIP-master"))
%addpath(genpath("/Users/???/MATLAB/sedumi-master"))
%addpath(genpath("/Library/gurobi1001/macos_universal2/matlab/"))

d = 10; % dimension
sample_sizes = [10, 20, 50, 75, 100];
T = 20; % iterations
rho = 0.1; % Wp perturbation size (here, will be a translation)
eps = 0.1; % TV perturbation size
sigma = sqrt(d); % moment bound (sqrt(d) * covariance)
contamination_scale_factor = 100; % push to make standard WDRO have unbounded excess risk

dual_norm = 2;
verbose = 0; % set to 1 to see the solver's progress

%% compute excess risks

standard_excess_risks = zeros(T,length(sample_sizes));
outlier_robust_excess_risks = zeros(T,length(sample_sizes));

for t = 1:T
    disp("iteration:")
    disp(t)
    n = sample_sizes(end);
    X = randn(n,d); % true features, each row is one sample
    theta_star = randn(d,1); % coefficients for linear hypothesis
    theta_star = theta_star./sqrt(theta_star'*theta_star); % normalize to unit vector
    y = sign(X * theta_star);

    X_tilde = X;
    % TV perturbation
    X_tilde(1:floor(eps*n),:) = X(1:floor(eps*n),:)*contamination_scale_factor;
    y_tilde(1:floor(eps*n)) = -y(1:floor(eps*n));
    % Wp perturbation
    translation = zeros(1,d);
    translation(1) = rho;
    X_tilde = X_tilde + repmat(translation,n,1); % translate all feature vectors by rho
    
    % risk of true coefficients
    best_risk = 0;
    
    for i = 1:length(sample_sizes)
        m = sample_sizes(i);
        disp("sample size")
        disp(m)
        % using standard WDRO on corrupted data
        disp("standard DRO excess risk")
        [theta_hat, ~] = regular_WDRO_classification(X_tilde(1:m,:), y_tilde(1:m), rho, dual_norm, verbose);
        standard_excess_risk = mean(max(0,1 - y.*(X*theta_hat))) - best_risk;
        disp(standard_excess_risk)
        standard_excess_risks(t,i) = standard_excess_risk;
        
        % using outlier-robust WDRO
        disp("outlier-robust DRO excess risk")
        x_0 = cheap_robust_mean_estimate(X_tilde, 2*eps);
        [theta_hat2, ~] = outlier_robust_WDRO_classification(X_tilde(1:m,:), y_tilde(1:m), sigma, rho, 2*eps, x_0', dual_norm, verbose);
        outlier_robust_excess_risk = mean(max(0,1 - y.*(X*theta_hat2))) - best_risk;
        disp(outlier_robust_excess_risk)
        outlier_robust_excess_risks(t,i) = outlier_robust_excess_risk;
    end
end

%% save data

save(strcat("exp3_classification_",datestr(now)),"standard_excess_risks", "outlier_robust_excess_risks")

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

title("\rm Excess Classification Risk with WDRO")
ax = gca;
ax.TitleFontSizeMultiplier = 1.5;
xl = xlabel("# samples");
fontsize(xl,'increase')
fontsize(xl,'increase')
xlim([10,100])
yl = ylabel("excess risk (hinge loss)");
fontsize(yl,'increase')
fontsize(yl,'increase')
leg = legend("standard WDRO","OR-WDRO");
fontsize(leg,'increase')
fontsize(leg,'increase')
hold off