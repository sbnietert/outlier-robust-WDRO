%% setup

clc
clearvars
%addpath(genpath("/Users/???/MATLAB/YALMIP-master"))
%addpath(genpath("/Users/???/MATLAB/sedumi-master"))
%addpath(genpath("/Library/gurobi1001/macos_universal2/matlab/"))

dimensions = [5, 10, 25, 40];
n = 20;
T = 20; % iterations
rho = 0.1; % Wp perturbation size (here, will be a translation)
eps = 0.05; % TV perturbation size
cov_bound = 2;
contamination_scale_factor = 100; % push to make standard WDRO have unbounded excess risk

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

dual_norm = 2;
verbose = 0; % set to 1 to see the solver's progress

%% compute excess risks

standard_excess_risks = zeros(T,length(dimensions));
outlier_robust_excess_risks = zeros(T,length(dimensions));
outlier_robust_cov_excess_risks = zeros(T,length(dimensions));

for i = 1:length(dimensions)
    disp("dimension:")
    d = dimensions(i);
    disp(d)
    sigma = sqrt(d)*cov_bound;

    theta0 = zeros(d,1); % coefficients for linear hypothesis
    theta0(1) = 1; % set to standard basis vector for simplicity
    theta1 = zeros(d,1);
    theta1(1) = cos(2*asin(rho/2*sqrt(pi/2)));
    theta1(2) = sin(2*asin(rho/2*sqrt(pi/2)));
    % norm(theta0 - theta1) = rho * sqrt(pi/2)
    theta_tilde = -theta1 * contamination_scale_factor;

    for t = 1:T
        disp("iteration:")
        disp(t)

        X = randn(n,d); % true features, each row is one sample
        y = X * theta0;
        X_tilde = X;
        X_tilde(1:floor(eps*n),:) = X_tilde(1:floor(eps*n),:)*contamination_scale_factor;
        y_tilde = X_tilde * theta1;
        y_tilde(1:floor(eps*n)) = X_tilde(1:floor(eps*n),:) * theta_tilde; % use theta_tilde for eps fraction of labels
        
        % risk of true coefficients
        best_risk = mean(abs(X * theta0 - y)); % = 0 for current setup

        % using standard WDRO on corrupted data
        disp("standard DRO excess risk")
        [theta_star, ~] = regular_WDRO(X_tilde, y_tilde, rho, dual_norm, verbose);
        standard_excess_risks(t,i) = mean(abs(X * theta_star - y)) - best_risk;
        disp(standard_excess_risks(t,i))

        % cheap robust mean estimate
        z_0 = cheap_robust_mean_estimate(X_tilde, 2*eps);
        z_0(end+1) = cheap_robust_mean_estimate(y_tilde, 2*eps);
        z_0 = z_0';

        % using outlier-robust WDRO with G2 class
        disp("outlier-robust DRO excess risk (estimated mean)")

        [theta_star, ~] = outlier_robust_WDRO(X_tilde, y_tilde, sigma, rho, 0, z_0, dual_norm, verbose);
        outlier_robust_excess_risks(t,i) = mean(abs(X * theta_star - y)) - best_risk;
        disp(outlier_robust_excess_risks(t,i));

        % using outlier-robust WDRO with covariance class
        disp("outlier-robust DRO excess risk, cov (estimated mean)")
        [theta_star, ~] = outlier_robust_WDRO_cov(X_tilde, y_tilde, cov_bound, rho, 0, z_0, dual_norm, verbose);
        outlier_robust_cov_excess_risks(t,i) = mean(abs(X * theta_star - y)) - best_risk;
        disp(outlier_robust_cov_excess_risks(t,i));
    end
end

%% save data
save(strcat("exp2_",datestr(now)),"outlier_robust_excess_risks", "outlier_robust_cov_excess_risks")

%% bootstrapping

K = 100; % number of bootstrap resamples
outlier_robust_averages = mean(outlier_robust_excess_risks,1);
outlier_robust_cov_averages = mean(outlier_robust_cov_excess_risks,1);

outlier_robust_bootstraps = zeros(K, length(dimensions));
outlier_robust_cov_bootstraps = zeros(K, length(dimensions));
for k = 1:K
    for i = 1:length(dimensions)
        d = dimensions(i);
        outlier_robust_bootstraps(k,i) = mean(randsample(outlier_robust_excess_risks(:,i),T,true));
        outlier_robust_cov_bootstraps(k,i) = mean(randsample(outlier_robust_cov_excess_risks(:,i),T,true));
    end
end
outlier_robust_bootstraps = sort(outlier_robust_bootstraps,1);
outlier_robust_cov_bootstraps = sort(outlier_robust_cov_bootstraps,1);
outlier_robust_p10 = outlier_robust_bootstraps(round(K*.1),:);
outlier_robust_p90 = outlier_robust_bootstraps(round(K*.9),:);
outlier_robust_cov_p10 = outlier_robust_cov_bootstraps(round(K*.1),:);
outlier_robust_cov_p90 = outlier_robust_cov_bootstraps(round(K*.9),:);

%% plots
hold on
errorbar(dimensions, outlier_robust_averages, outlier_robust_averages - outlier_robust_p10, outlier_robust_p90 - outlier_robust_averages);
errorbar(dimensions, outlier_robust_cov_averages, outlier_robust_cov_averages - outlier_robust_cov_p10, outlier_robust_cov_p90 - outlier_robust_cov_averages);

title("\rm Excess Risk for Varied Dimension and Method")
ax = gca;
ax.TitleFontSizeMultiplier = 1.5;
xlabel("dimension")
ylabel("excess risk (mean absolute deviation)")
legend("outlier-robust WDRO w/ A = G_2","outlier-robust WDRO w/ A = G_{cov}")
hold off