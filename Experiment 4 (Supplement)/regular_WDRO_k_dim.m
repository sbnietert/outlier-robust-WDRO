function [theta_star, obj_star] = regular_WDRO_k_dim(X, Y, rho, dual_norm, verbose)
% This implementation is for p = 1, and Z = R^d

    % Define parameters
    [n, d_x] = size(X);
    d_y = size(Y, 2);
    Z = [X, Y];
    d = d_x + d_y;
    J = 2 ^ d_y;
    p = 1; 
    q = 2;  
    
    % Define variables
    lambda_2 = sdpvar(1);
    s = sdpvar(n, 1);
    zeta_W = sdpvar(d, n, J);
    theta = sdpvar(d_x, d_y, 'full');
    
    % Define objective function
    objective =lambda_2 * rho^p + mean(s);
    
    % Define constraints
    constraints = cell(n*J+1, 1);
    constraints{n*J+1} = [lambda_2 >= 0, s >= 0];
    counter = 0;
    for i = 1 : n
        for j = 1 : J
            piece = mod(counter,J) + 1;
            str = dec2bin(piece - 1, d_y);
            coeff = ones(d_y, 1);
            coeff(str == '1') = -1;
            counter = counter + 1;
            constraints{counter} = [s(i) >= Z(i,:) * zeta_W(:,i, piece);
                                    [theta * coeff; -coeff] + zeta_W(:, i, piece) == 0;
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