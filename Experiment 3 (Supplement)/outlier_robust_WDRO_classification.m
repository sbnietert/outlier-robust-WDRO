function [theta_star, obj_star] = outlier_robust_WDRO_classification(X, y, sigma, rho, vareps, x_0, dual_norm, verbose)
% This implementation is for p = 1, q = 2, and Z = R^d

    % Define parameters
    [n, d] = size(X);
    J = 2;
    
    % Define variables
    lambda_1 = sdpvar(1);
    lambda_2 = sdpvar(1);
    alpha = sdpvar(1);
    s = sdpvar(n, 1);
    zeta_G = sdpvar(d, n, J);
    zeta_W = sdpvar(d, n, J);
    tau = sdpvar(1, n, J);
    theta = sdpvar(d, 1);
    
    % Define objective function
    objective = lambda_1 * sigma^2 + lambda_2 * rho + 1 / (n * (1 - vareps)) * sum(s) + alpha;
    
    % Define constraints
    constraints = cell(n+1, 1);
    constraints{n+1} = [lambda_1 >= 0, lambda_2 >= 0, s >= 0, tau >= 0];
    for i = 1 : n
        constraints{i} = [s(i) >= 1 + x_0' * zeta_G(:, i, 1) + tau(:, i, 1) + X(i,:) * zeta_W(:,i, 1) - alpha;
                          s(i) >=  x_0' * zeta_G(:, i, 2) + tau(:, i, 2) + X(i,:) * zeta_W(:,i, 2) - alpha;
                          y(i) * theta + zeta_G(:, i, 1) + zeta_W(:, i, 1) == 0;
                          zeta_G(:, i, 2) + zeta_W(:, i, 2) == 0;
                          rcone(zeta_G(:, i, 1), lambda_1, 0.5 * tau(:, i, 1));
                          rcone(zeta_G(:, i, 2), lambda_1, 0.5 * tau(:, i, 2));
                          norm(zeta_W(:, i, 1), dual_norm) <= lambda_2;
                          norm(zeta_W(:, i, 2), dual_norm) <= lambda_2];
    end
    
    % Optimize model
    ops = sdpsettings('solver', 'sedumi', 'verbose', verbose);
    optimize([constraints{:}], objective, ops);
    
    % Return the solution
    theta_star = value(theta);
    obj_star = value(objective);
    
end