function [theta_star, obj_star] = outlier_robust_WDRO_cov(X, y, sigma, rho, vareps, z_0, dual_norm, verbose)
% This implementation is for p = 1, q = 2, and Z = R^d

    % Define parameters
    Z = [X, y(:)];
    [n, d] = size(Z);
    J = 2;
    p = 1; 
    q = 2; 
    
    % Define variables
    Lambda_1 = sdpvar(d, d, 'symmetric');
    lambda_2 = sdpvar(1);
    s = sdpvar(n, 1);
    zeta_G = sdpvar(d, n, J);
    zeta_W = sdpvar(d, n, J);
    tau = sdpvar(1, n, J);
    theta = sdpvar(d-1, 1);
    
    % Define objective function
    objective = z_0' * Lambda_1 * z_0 + sigma^q * trace(Lambda_1) + lambda_2 * rho^p + 1 / (n * (1 - vareps)) * sum(s);
    
    % Define constraints
    constraints = cell(n+1, 1);
    constraints{n+1} = [Lambda_1 >= 0, lambda_2 >= 0, s >= 0, tau >= 0];
    for i = 1 : n
        constraints{i} = [s(i) >= z_0' * zeta_G(:, i, 1) + tau(:, i, 1) + Z(i,:) * zeta_W(:,i, 1);
                          s(i) >= z_0' * zeta_G(:, i, 2) + tau(:, i, 2) + Z(i,:) * zeta_W(:,i, 2);
                          [-theta; 1] + zeta_G(:, i, 1) + zeta_W(:, i, 1) == 0;
                          [theta; -1] + zeta_G(:, i, 2) + zeta_W(:, i, 2) == 0;
                          [Lambda_1, zeta_G(:, i, 1); zeta_G(:, i, 1)', 4*tau(:, i, 1)] >= 0;
                          [Lambda_1, zeta_G(:, i, 2); zeta_G(:, i, 2)', 4*tau(:, i, 2)] >= 0;
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