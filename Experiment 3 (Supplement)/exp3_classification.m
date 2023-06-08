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

theta = randn(d,1); % coefficients for affine hypothesis
theta = theta./sqrt(theta'*theta); % normalize to unit vector

%% compute excess risks

standard_excess_risks = zeros(T,length(sample_sizes));
outlier_robust_0_excess_risks = zeros(T,length(sample_sizes));

for t = 1:T
    disp("iteration:")
    disp(t)
    n = sample_sizes(end);
    X = randn(n,d); % true features, each row is one sample
    theta = randn(d,1); % coefficients for linear hypothesis
    theta = theta./sqrt(theta'*theta); % normalize to unit vector
    y = sign(X * theta);

    theta_tilde = contamination_scale_factor * -theta; % large coefficient vector opposite to theta
    translation = zeros(1,d);
    translation(1) = rho;
    X_tilde = X + repmat(translation,n,1); % translate all feature vectors by rho
    X_tilde(1:floor(eps*n),:) = X_tilde(1:floor(eps*n),:)*contamination_scale_factor;
    y_tilde = y;
    y_tilde(1:floor(eps*n)) = X_tilde(1:floor(eps*n),:) * theta_tilde; % use theta_tilde for eps fraction of labels
    y_tilde(1:floor(eps*n)) = sign(X_tilde(1:floor(eps*n),:) * theta_tilde); % use theta_tilde for eps fraction of labels
    
    % risk of true coefficients
    best_risk = 0;
    
    for i = 1:length(sample_sizes)
        m = sample_sizes(i);
        disp("sample size")
        disp(m)
        % using standard WDRO on corrupted data
        disp("standard DRO excess risk")
        [theta_star, obj_star] = regular_WDRO_classification(X_tilde(1:m,:), y_tilde(1:m), rho, dual_norm, verbose);
        standard_excess_risk = mean(max(0,1 - y.*(X*theta_star))) - best_risk;
        disp(standard_excess_risk)
        standard_excess_risks(t,i) = standard_excess_risk;
        
        % using outlier-robust WDRO
        disp("outlier-robust DRO excess risk (eps = 0)")
        x_0 = cheap_robust_mean_estimate(X_tilde, 2*eps);
        [theta_star, obj_star2] = outlier_robust_WDRO_classification(X_tilde(1:m,:), y_tilde(1:m), sigma, rho, 0, x_0', dual_norm, verbose);
        outlier_robust_excess_risk = mean(max(0,1 - y.*(X*theta_star))) - best_risk;
        disp(outlier_robust_excess_risk)
        outlier_robust_0_excess_risks(t,i) = outlier_robust_excess_risk;
    end
end

%% save data
save(strcat("exp4_",datestr(now)),"standard_excess_risks", "outlier_robust_0_excess_risks")%, "outlier_robust_1_excess_risks", "outlier_robust_2_excess_risks")

%% load data
load

%% bootstrapping


% d = 10; % dimension
% sample_sizes = [10, 20, 50, 75, 100];%, 200, 500];
% T = 20; % iterations
% rho = 0.1;

K = 100; % number of bootstrap resamples
standard_averages = mean(standard_excess_risks,1);
outlier_robust_0_averages = mean(outlier_robust_0_excess_risks,1);

standard_bootstraps = zeros(K, length(sample_sizes));
outlier_robust_0_bootstraps = zeros(K, length(sample_sizes));
for k = 1:K
    for i = 1:length(sample_sizes)
        n = sample_sizes(i);
        standard_bootstraps(k,i) = mean(randsample(standard_excess_risks(:,i),T,true));
        outlier_robust_0_bootstraps(k,i) = mean(randsample(outlier_robust_0_excess_risks(:,i),T,true));
    end
end
standard_bootstraps = sort(standard_bootstraps,1);
outlier_robust_0_bootstraps = sort(outlier_robust_0_bootstraps,1);
standard_p10 = standard_bootstraps(round(K*.1),:);
standard_p90 = standard_bootstraps(round(K*.9),:);
outlier_robust_0_p10 = outlier_robust_0_bootstraps(round(K*.1),:);
outlier_robust_0_p90 = outlier_robust_0_bootstraps(round(K*.9),:);

%% plots
hold on
errorbar(sample_sizes, standard_averages, standard_averages - standard_p10, standard_p90 - standard_averages);
errorbar(sample_sizes, outlier_robust_0_averages, outlier_robust_0_averages - outlier_robust_0_p10, outlier_robust_0_p90 - outlier_robust_0_averages);

title("\rm Excess Classification Risk with WDRO")
ax = gca;
ax.TitleFontSizeMultiplier = 1.5;
xlabel("# samples")
xlim([10,100])
ylabel("excess risk (hinge loss)")
legend("standard WDRO","outlier-robust WDRO")
hold off