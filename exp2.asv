clc
clearvars
%addpath(genpath("/Users/sloan/MATLAB/YALMIP-master"))
%addpath(genpath("/Users/sloan/MATLAB/sedumi-master"))
%addpath(genpath("/Library/gurobi1001/macos_universal2/matlab/"))

dimensions = [5, 10, 25, 40];
n = 20;
T = 20; % iterations
rho = 0.1; % Wp perturbation size (here, will be a translation)
eps = 0.05; % TV perturbation size
sigma = 2; % moment bound, specific to distributions here
contamination_scale_factor = 100; % push to make standard WDRO have unbounded excess risk

dual_norm = 2;
verbose = 0; % you can set it to 1 if you want to see the solver's progress

standard_excess_risks = zeros(T,length(dimensions));
outlier_robust_excess_risks = zeros(T,length(dimensions));
outlier_robust_cov_excess_risks = zeros(T,length(dimensions));


for i = 1:length(dimensions)
    disp("dimension:")
    d = dimensions(i);
    disp(d)

    theta = randn(d,1); % coefficients for affine hypothesis
    theta = theta./sqrt(theta'*theta); % normalize to unit vector
    theta_0 = randn(1); % bias for affine hypothesis
    
    
    for t = 1:T
        disp("iteration:")
        disp(t)
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
        
        % using standard WDRO on corrupted data
        disp("standard DRO excess risk")
        [theta_star, theta_0_star, obj_star] = regular_WDRO(X_tilde, y_tilde, 0.001, dual_norm, verbose);
        standard_excess_risk = mean(abs(X * theta_star + theta_0_star - y)) - best_risk;
        disp(standard_excess_risk)
        standard_excess_risks(t,i) = standard_excess_risk;

        % using outlier-robust WDRO, proper sigma, estimated mean as
        % center
        disp("outlier-robust DRO excess risk (estimated mean)")
        z_0 = cheap_robust_mean_estimate(X_tilde, 2*eps);
        z_0(end+1) = cheap_robust_mean_estimate(y_tilde, 2*eps);
        z_0 = z_0';
        [theta_star2, theta_0_star2, obj_star2] = outlier_robust_WDRO(X_tilde, y_tilde, sigma*sqrt(d), rho, 0, z_0, dual_norm, verbose);
        outlier_robust_excess_risk = mean(abs(X * theta_star2 + theta_0_star2 - y)) - best_risk;
        disp(outlier_robust_excess_risk)
        outlier_robust_excess_risks(t,i) = outlier_robust_excess_risk;

        % using outlier-robust WDRO with covariance class, proper sigma, estimated mean as
        % center
        disp("outlier-robust DRO excess risk, cov (estimated mean)")
        [theta_star3, theta_0_star3, obj_star3] = WDRO_cov(X_tilde, y_tilde, sigma, rho, 0, z_0, dual_norm, verbose);
        outlier_robust_cov_excess_risk = mean(abs(X * theta_star3 + theta_0_star3 - y)) - best_risk;
        disp(outlier_robust_cov_excess_risk)
        outlier_robust_cov_excess_risks(t,i) = outlier_robust_cov_excess_risk;
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
errorbar(dimensions, outlier_robust_averages + rho, outlier_robust_averages - outlier_robust_p10, outlier_robust_p90 - outlier_robust_averages);
errorbar(dimensions, outlier_robust_cov_averages + rho, outlier_robust_cov_averages - outlier_robust_cov_p10, outlier_robust_cov_p90 - outlier_robust_cov_averages);

title("\rm Excess Risk for Varied Dimension and Method")
ax = gca;
ax.TitleFontSizeMultiplier = 1.5;
xlabel("dimension")
%ylim([0,3])
%xlim([10,100])
ylabel("excess risk (mean absolute deviation)")
legend("outlier-robust WDRO w/ A = G_2","outlier-robust WDRO w/ A = G_{cov}")
hold off

%% plotting
standard_averages = mean(standard_excess_risks,1);
standard_stds = std(standard_excess_risks,1);
outlier_robust_averages = mean(outlier_robust_excess_risks,1);
outlier_robust_stds = std(outlier_robust_excess_risks,1);
outlier_robust_cov_averages = mean(outlier_robust_cov_excess_risks,1);
outlier_robust_cov_stds = std(outlier_robust_cov_excess_risks,1);
%errorbar(dimensions, standard_averages,standard_stds)
hold on
errorbar(dimensions, outlier_robust_averages,outlier_robust_stds)
errorbar(dimensions, outlier_robust_cov_averages,outlier_robust_cov_stds)
xlabel("dimension d")
ylabel("excess risk")
legend("outlier-robust WDRO, G2","outlier-robust WDRO, cov")
hold off