function [theta_star, obj_star] = regular_WDRO_classification_v2(X, y, rho, dual_norm, verbose)
% This implementation is for p = 1, and Z = R^d

    % Define parameters
    [n, d] = size(X);
    
    % Define variables
    s = sdpvar(n, 1);
    theta = sdpvar(d, 1);
    
    % Define objective function
    objective = norm(theta, dual_norm) * rho + mean(s);
    
    % Define constraints
    constraints = cell(n, 1);
    for i = 1 : n
        constraints{i} = [s(i) >= 1 - y(i) * X(i,:) * theta;
                          s(i) >= 0];
    end
    
    % Optimize model
    ops = sdpsettings('solver', 'sedumi', 'verbose', verbose);
    optimize([constraints{:}], objective, ops);
    
    % Return the solution
    theta_star = value(theta);
    obj_star = value(objective);
    
end