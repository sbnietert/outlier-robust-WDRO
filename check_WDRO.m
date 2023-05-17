clc
clearvars
%addpath(genpath("/Users/sloan/MATLAB/YALMIP-master"))
%addpath(genpath("/Users/sloan/MATLAB/sedumi-master"))
%addpath(genpath("/Library/gurobi1001/macos_universal2/matlab/"))

X = randn(100, 10);
theta = rand(10,1);
theta_0 = rand(1);
y = X * theta + theta_0 + 0.01 * randn(100, 1);
sigma = 0.1;
rho = 0.001;
vareps = 0.05;
z_0 = zeros(11, 1);
dual_norm = 2;
verbose = 1; % you can set it to 1 if you want to see the solver's progress

[theta_star, theta_0_star, obj_star] = WDRO(X, y, sigma, rho, vareps, z_0, dual_norm, verbose);
disp(norm(theta - theta_star))
disp(abs(theta_0 - theta_0_star))
disp(obj_star)