function llh = synthetic_log_likelihood(theta,S_obs,n,N)
% estimate the log of synthetic likelihood 
% theta:        vector of model parameters
% S_obs:        summary statistics from the observed data
% n:            the size of observed data
% N:            number of datsets generated to estimate the synmethic
%               likelihood

d = length(S_obs);
summary_stat = zeros(N,d);
for i = 1:N
    y = stblrnd(theta(1),theta(2),theta(3),theta(4),n);
    S = zeros(4,1);
    [S(1),S(2),S(3),S(4)] = stablecull(y);
    summary_stat(i,:) = S;
end
mu_hat = mean(summary_stat)';
Sigma_hat = cov(summary_stat);
Sigma_hat = single(Sigma_hat);

llh = -1/2*log(det(Sigma_hat))-(N-d-2)/2/(N-1)*(S_obs-mu_hat)'*(Sigma_hat\(S_obs-mu_hat));

end