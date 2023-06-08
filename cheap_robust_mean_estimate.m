% X : n x d matrix of n vectors in R^d
% eps : contamination fraction
function est = cheap_robust_mean_estimate(X, eps)
    [n,~] = size(X);
    X_sorted = sort(X,1);

    trim_start = ceil(eps*n);
    trim_end = floor((1-eps)*n);
    X_trimmed = X_sorted(trim_start:trim_end,:);
    est = mean(X_trimmed,1);
end