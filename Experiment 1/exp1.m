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
eps = 0.05; % TV perturbation size
sigma = sqrt(d); % moment bound (sqrt(d) * covariance)
contamination_scale_factor = 20;

% model for clean data:
%   X ~ N(0,I_d)
%   Y = X^T \theta_0
% model for contaminated data:
%   \tilde{X} = X w.p. 1-eps, = contamination_scale_factor * X w.p. eps
%   \tilde{Y} = - \tilde{X}^T \theta_1
%   ||\theta_0 - \theta_1|| = \rho*sqrt{pi/2}
% by spherical symmetry, Wp^eps(clean,contaminated) <= rho
% excess risk of decision \theta:
%   \sqrt{2/\pi} ||\theta - \theta_0||, by spherical symmetry

theta0 = zeros(d,1); % coefficients for linear hypothesis
theta0(1) = 1; % set to standard basis vector for simplicity
theta1 = zeros(d,1);
theta1(1) = cos(2*asin(rho/2*sqrt(pi/2)));
theta1(2) = sin(2*asin(rho/2*sqrt(pi/2)));
% norm(theta0 - theta1) = rho * sqrt(pi/2)
theta_tilde = -theta1;


dual_norm = 2; % \ell_2 norm
verbose = 0; % set to 1 to see the solver's progress

%% compute excess risks

standard_excess_risks = zeros(T,length(sample_sizes));
outlier_robust_0_excess_risks = zeros(T,length(sample_sizes));
outlier_robust_1_excess_risks = zeros(T,length(sample_sizes));
outlier_robust_2_excess_risks = zeros(T,length(sample_sizes));

for t = 1:T
    disp("iteration:")
    disp(t)

    n = sample_sizes(end);
    X = randn(n,d); % true features, each row is one sample
    y = X * theta0;
    X_tilde = X;
    X_tilde(1:floor(eps*n),:) = X_tilde(1:floor(eps*n),:)*contamination_scale_factor;
    y_tilde = X_tilde * theta1;
    y_tilde(1:floor(eps*n)) = X_tilde(1:floor(eps*n),:) * theta_tilde; % use theta_tilde for eps fraction of labels

    % risk of true coefficients
    best_risk = mean(abs(X * theta0 - y)); % = 0 for current setup

    for i = 1:length(sample_sizes)
        m = sample_sizes(i);
        disp("sample size")
        disp(m)
        % using standard WDRO on corrupted data
        disp("standard DRO excess risk")
        [theta_star, ~] = regular_WDRO(X_tilde(1:m,:), y_tilde(1:m), rho, dual_norm, verbose);
        standard_excess_risks(t,i) = mean(abs(X * theta_star - y)) - best_risk;
        disp(standard_excess_risks(t,i))


        % get cheap robust mean estimate
        z_0 = cheap_robust_mean_estimate(X_tilde, 2*eps);
        z_0(end+1) = cheap_robust_mean_estimate(y_tilde, 2*eps);
        z_0 = z_0';
        
        % using outlier-robust WDRO, proper sigma, eps = 0
        disp("outlier-robust DRO excess risk (eps = 0)");
        [theta_star, ~] = outlier_robust_WDRO(X_tilde(1:m,:), y_tilde(1:m), sigma, rho, 0, z_0, dual_norm, verbose);
        outlier_robust_0_excess_risks(t,i) = mean(abs(X * theta_star - y)) - best_risk;
        disp(outlier_robust_0_excess_risks(t,i))

        % using outlier-robust WDRO, proper sigma, eps
        disp("outlier-robust DRO excess risk (eps)")
        [theta_star, ~] = outlier_robust_WDRO(X_tilde(1:m,:), y_tilde(1:m), sigma, rho, eps, z_0, dual_norm, verbose);
        outlier_robust_1_excess_risks(t,i) = mean(abs(X * theta_star - y)) - best_risk;
        disp(outlier_robust_1_excess_risks(t,i))

        % using outlier-robust WDRO, proper sigma, 2eps
        disp("outlier-robust DRO excess risk (2eps)")
        [theta_star2, ~] = outlier_robust_WDRO(X_tilde(1:m,:), y_tilde(1:m), sigma, rho, 2*eps, z_0, dual_norm, verbose);
        outlier_robust_2_excess_risks(t,i) = mean(abs(X * theta_star - y)) - best_risk;
        disp(outlier_robust_2_excess_risks(t,i))
    end
end

%% save data
save(strcat("exp1_",datestr(now)),"standard_excess_risks", "outlier_robust_0_excess_risks", "outlier_robust_1_excess_risks", "outlier_robust_2_excess_risks")

%% bootstrapping

% d = 10; % dimension
% sample_sizes = [10, 20, 50, 75, 100];%, 200, 500];
% T = 20; % iterations
% rho = 0.1;

K = 100; % number of bootstrap resamples
standard_averages = mean(standard_excess_risks,1);
outlier_robust_0_averages = mean(outlier_robust_0_excess_risks,1);
outlier_robust_1_averages = mean(outlier_robust_1_excess_risks,1);
outlier_robust_2_averages = mean(outlier_robust_2_excess_risks,1);

standard_bootstraps = zeros(K, length(sample_sizes));
outlier_robust_0_bootstraps = zeros(K, length(sample_sizes));
outlier_robust_1_bootstraps = zeros(K, length(sample_sizes));
outlier_robust_2_bootstraps = zeros(K, length(sample_sizes));
for k = 1:K
    for i = 1:length(sample_sizes)
        n = sample_sizes(i);
        standard_bootstraps(k,i) = mean(randsample(standard_excess_risks(:,i),T,true));
        outlier_robust_0_bootstraps(k,i) = mean(randsample(outlier_robust_0_excess_risks(:,i),T,true));
        outlier_robust_1_bootstraps(k,i) = mean(randsample(outlier_robust_1_excess_risks(:,i),T,true));
        outlier_robust_2_bootstraps(k,i) = mean(randsample(outlier_robust_2_excess_risks(:,i),T,true));
    end
end
standard_bootstraps = sort(standard_bootstraps,1);
outlier_robust_0_bootstraps = sort(outlier_robust_0_bootstraps,1);
outlier_robust_1_bootstraps = sort(outlier_robust_0_bootstraps,1);
outlier_robust_2_bootstraps = sort(outlier_robust_0_bootstraps,1);
standard_p10 = standard_bootstraps(round(K*.1),:);
standard_p90 = standard_bootstraps(round(K*.9),:);
outlier_robust_0_p10 = outlier_robust_0_bootstraps(round(K*.1),:);
outlier_robust_0_p90 = outlier_robust_0_bootstraps(round(K*.9),:);
outlier_robust_1_p10 = outlier_robust_1_bootstraps(round(K*.1),:);
outlier_robust_1_p90 = outlier_robust_1_bootstraps(round(K*.9),:);
outlier_robust_2_p10 = outlier_robust_2_bootstraps(round(K*.1),:);
outlier_robust_2_p90 = outlier_robust_2_bootstraps(round(K*.9),:);

%% plots
hold on
errorbar(sample_sizes, standard_averages, standard_averages - standard_p10, standard_p90 - standard_averages);
errorbar(sample_sizes, outlier_robust_0_averages, outlier_robust_0_averages - outlier_robust_0_p10, outlier_robust_0_p90 - outlier_robust_0_averages);
errorbar(sample_sizes, outlier_robust_1_averages, outlier_robust_1_averages - outlier_robust_1_p10, outlier_robust_1_p90 - outlier_robust_1_averages);
errorbar(sample_sizes, outlier_robust_2_averages, outlier_robust_2_averages - outlier_robust_2_p10, outlier_robust_2_p90 - outlier_robust_2_averages);

title("\rm Excess Risk for Varied Sample Size and Method")
ax = gca;
ax.TitleFontSizeMultiplier = 1.5;
xlabel("# samples")
%ylim([0,3])
xlim([10,100])
ylabel("excess risk (mean absolute deviation)")
legend("standard WDRO","outlier-robust WDRO w/ ε = 0", "outlier-robust WDRO w/ ε = ε₀", "outlier-robust WDRO w/ ε = 2ε₀")
hold off