function [theta_star, obj_star] = outlier_robust_WDRO_k_dim(X, Y, sigma, rho, vareps, z_0, dual_norm, verbose)
% This implementation is for p = 1, q = 2, and Z = R^d

    % Define parameters
    [n, d_x] = size(X);
    d_y = size(Y, 2);
    Z = [X, Y];
    d = d_x + d_y;
    J = 2 ^ d_y;
    p = 1; 
    q = 2;  
    
    % Define variables
    lambda_1 = sdpvar(1);
    lambda_2 = sdpvar(1);
    alpha = sdpvar(1);
    s = sdpvar(n, 1);
    zeta_G = sdpvar(d, n, J);
    zeta_W = sdpvar(d, n, J);
    tau = sdpvar(1, n, J);
    theta = sdpvar(d_x, d_y, 'full');
    
    % Define objective function
    objective = lambda_1 * sigma^q + lambda_2 * rho^p + 1 / (n * (1 - vareps)) * sum(s) + alpha;
    
    % Define constraints
    constraints = cell(n*J+1, 1);
    constraints{n*J+1} = [lambda_1 >= 0, lambda_2 >= 0, s >= 0, tau >= 0];
    counter = 0;
    for i = 1 : n
        for j = 1 : J
            piece = mod(counter,J) + 1;
            str = dec2bin(piece - 1, d_y);
            coeff = ones(d_y, 1);
            coeff(str == '1') = -1;
            counter = counter + 1;
            constraints{counter} = [s(i) >= z_0' * zeta_G(:, i, piece) + tau(:, i, piece) + Z(i,:) * zeta_W(:,i, piece) - alpha;
                                    [theta * coeff; -coeff] + zeta_G(:, i, piece) + zeta_W(:, i, piece) == 0;
                                    rcone(zeta_G(:, i, piece), lambda_1, 0.5 * tau(:, i, piece));
                                    norm(zeta_W(:, i, piece), dual_norm) <= lambda_2];
        end
    end
    
    % Optimize model
    ops = sdpsettings('solver', 'sedumi', 'verbose', verbose);
    optimize([constraints{:}], objective, ops);
    
    % Return the solution
    theta_star = value(theta);
    obj_star = value(objective);
    
end