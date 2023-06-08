function [theta_star, obj_star] = regular_WDRO_exp1(X, y, rho, dual_norm, verbose)
% This implementation is for p = 1, and Z = R^d

    % Define parameters
    Z = [X, y(:)];
    [n, d] = size(Z);
    p = 1; 
    
    % Define variables
    s = sdpvar(n, 1);
    theta = sdpvar(d-1, 1);
    
    % Define objective function
    objective = norm([theta; -1], dual_norm) * rho^p + mean(s);
    
    % Define constraints
    constraints = cell(n, 1);
    for i = 1 : n
        constraints{i} = [s(i) >= Z(i,:) * [theta; -1];
                          s(i) >= Z(i,:) * [-theta; 1]];
    end
    
    % Optimize model
    ops = sdpsettings('solver', 'sedumi', 'verbose', verbose);
    optimize([constraints{:}], objective, ops);
    
    % Return the solution
    theta_star = value(theta);
    obj_star = value(objective);
    
end