function [theta_star, theta_0_star, obj_star] = regular_WDRO(X, y, rho, dual_norm, verbose)
% This implementation is for p = 1, and Z = R^d

    % Define parameters
    Z = [X, y(:)];
    [n, d] = size(Z);
    J = 2;
    p = 1; 
    q = 2; 
    
    % Define variables
    s = sdpvar(n, 1);
    zeta_G = sdpvar(d, n, J);
    zeta_W = sdpvar(d, n, J);
    tau = sdpvar(1, n, J);
    theta = sdpvar(d-1, 1);
    theta_0 = sdpvar(1);
    
    % Define objective function
    objective = norm([theta; -1], dual_norm) * rho^p + mean(s);
    
    % Define constraints
    constraints = cell(n, 1);
    for i = 1 : n
        constraints{i} = [s(i) >= theta_0 + Z(i,:) * [theta; -1];
                          s(i) >= - theta_0 + Z(i,:) * [-theta; 1]];
    end
    
    % Optimize model
    ops = sdpsettings('solver', 'sedumi', 'verbose', verbose);
    optimize([constraints{:}], objective, ops);
    
    % Return the solution
    theta_star = value(theta);
    theta_0_star = value(theta_0);
    obj_star = value(objective);
    
end