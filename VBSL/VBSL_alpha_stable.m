%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% VBSL with Cholesky decomposition for alpha stable example 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Generating the observed data
clear all
rng(100); % fix random seed
alpha_true = 1.5; beta_true = 0.5; gamma_true = 1; delta_true = 0;
theta_true = [alpha_true beta_true gamma_true delta_true];
n = 200;
y_obs = stblrnd(alpha_true,beta_true,gamma_true,delta_true,n);
S_obs = zeros(4,1);
[S_obs(1),S_obs(2),S_obs(3),S_obs(4)] = stablecull(y_obs);

%% model setting
d = 4; % size of model parameters
mu_prior = zeros(d,1);
Sigma_prior = 100*eye(d);

%% VBSL setting
S = 200; % number of Monte Carlo samples to estimate the gradient of LB
N = 100; % number of synthetic datasets used to estimate the synthetic likelihood
dim = d+d*(d+1)/2; % dimension of variational parameter lambda
beta1_adap_weight = 0.9; % exponential smooth weight in adaptive learning
beta2_adap_weight = 0.9; % exponential smooth weight in adaptive learning
eps0 = 0.001; % fixed learning rate
max_iter = 5000; % maximum interation allowed
tau_threshold = max_iter/2; % threshold to start reduce the fixed learning rate
t_w = 50; % window size for smoothing LB
patience_max = t_w; % max patience 
norm_gradient_threshold = 100; % norm in gradient clipping

%% Initialization
mu = zeros(d,1); % initial value for mu
L = 10*eye(d); % lower triangular matrix in the Cholesky decomposition
lambda = [mu;vech(L)]; % initial lambda

% STAGE 1: prepare for the main VB iteration.
Sigma_inv = L*L'; Sigma = eye(d)/Sigma_inv; L_inv = eye(d)/L;
cv = zeros(length(lambda),1); % control variate, initilised to be all zero
h_lambda = zeros(S,1); % h_lambda function 
gra_log_q_lambda = zeros(S,dim); % gradient of log_q w.r.t lambda
grad_log_q_h_function = zeros(S,dim); % (gradient of log_q) x h_lambda(theta); used in calculating control variates 
grad_log_q_h_function_cv = zeros(S,dim); % control_variate version: (gradient of log_q) x (h_lambda(theta)-c)
theta_samples = mvnrnd(mu,Sigma,S); % generate theta samples
parfor s = 1:S  
%parfor s = 1:S    
    theta_tilde = theta_samples(s,:)';
    % we now transform unconstrained theta_tilde to original theta
    alpha_tilde = theta_tilde(1); beta_tilde = theta_tilde(2); gamma_tilde = theta_tilde(3); delta = theta_tilde(4);
    alpha = (2*exp(alpha_tilde)+1.1)/(1+exp(alpha_tilde));
    beta = (exp(beta_tilde)-1)/(exp(beta_tilde)+1);
    gamma = exp(gamma_tilde);
    theta = [alpha;beta;gamma;delta];    
    llh = synthetic_log_likelihood(theta,S_obs,n,N); % synthetic log-likelihood

    log_prior = log(mvnpdf(theta_tilde,mu_prior,Sigma_prior));        
    log_q = log(mvnpdf(theta_tilde,mu,Sigma));
    gra_log_q_lambda(s,:) = [Sigma_inv*(theta_tilde-mu);vech(diag(1./diag(L))-(theta_tilde-mu)*(theta_tilde-mu)'*L)]';
    
    h_lambda(s) = log_prior+llh-log_q;
    grad_log_q_h_function(s,:) = gra_log_q_lambda(s,:)*h_lambda(s);    
    grad_log_q_h_function_cv(s,:) = gra_log_q_lambda(s,:).*(h_lambda(s)-cv');
end
% calculate the control variates 
for i = 1:dim
    aa = cov(grad_log_q_h_function(:,i),gra_log_q_lambda(:,i));
    cv(i) = aa(1,2)/aa(2,2);
end
grad_LB = mean(grad_log_q_h_function_cv)'; % estimate of the gradient of LB
LB = mean(h_lambda); % lower bound estimate

% gradient clipping
grad_norm = norm(grad_LB);
if norm(grad_LB)>norm_gradient_threshold
    grad_LB = (norm_gradient_threshold/grad_norm)*grad_LB;
end

% initialize adaptive learning 
g_adaptive = grad_LB; v_adaptive = g_adaptive.^2; 
g_bar_adaptive = g_adaptive; v_bar_adaptive = v_adaptive; 

% STAGE 2: VB iterations
iter = 1;
stop = false;
LB_bar = 0; patience = 0;
lambda_best = lambda;
while ~stop    
    
    mu = lambda(1:d);
    L = vechinv(lambda(d+1:end),2);
  
    h_lambda = zeros(S,1);
    Sigma_inv = L*L'; Sigma = eye(d)/Sigma_inv; L_inv = eye(d)/L;
    cv = zeros(length(lambda),1); % control variate, initilised to be all zero
    gra_log_q_lambda = zeros(S,dim); % gradient of log_q
    grad_log_q_h_function = zeros(S,dim); % (gradient of log_q) x h_lambda(theta) 
    grad_log_q_h_function_cv = zeros(S,dim); % control_variate version: (gradient of log_q) x (h_lambda(theta)-c)
    theta_samples = mvnrnd(mu,Sigma,S); 
    parfor s = 1:S    
        theta_tilde = theta_samples(s,:)';
        
        alpha_tilde = theta_tilde(1); beta_tilde = theta_tilde(2); gamma_tilde = theta_tilde(3); delta = theta_tilde(4);
        alpha = (2*exp(alpha_tilde)+1.1)/(1+exp(alpha_tilde));
        beta = (exp(beta_tilde)-1)/(exp(beta_tilde)+1);
        gamma = exp(gamma_tilde);
        theta = [alpha;beta;gamma;delta];
        llh = synthetic_log_likelihood(theta,S_obs,n,N);
        
        log_prior = log(mvnpdf(theta_tilde,mu_prior,Sigma_prior));
        log_q = log(mvnpdf(theta_tilde,mu,Sigma));
        gra_log_q_lambda(s,:) = [Sigma_inv*(theta_tilde-mu);vech(diag(1./diag(L))-(theta_tilde-mu)*(theta_tilde-mu)'*L)]';

        h_lambda(s) = log_prior+llh-log_q;
        grad_log_q_h_function(s,:) = gra_log_q_lambda(s,:)*h_lambda(s);    
        grad_log_q_h_function_cv(s,:) = gra_log_q_lambda(s,:).*(h_lambda(s)-cv');
    end
    for i = 1:dim
        aa = cov(grad_log_q_h_function(:,i),gra_log_q_lambda(:,i));
        cv(i) = aa(1,2)/aa(2,2);
    end
    grad_LB = mean(grad_log_q_h_function_cv)';

    % gradient clipping
    grad_norm = norm(grad_LB);    
    if norm(grad_LB)>norm_gradient_threshold
        grad_LB = (norm_gradient_threshold/grad_norm)*grad_LB;
    end
    
    % update adaptive learning 
    g_adaptive = grad_LB; v_adaptive = g_adaptive.^2; 
    g_bar_adaptive = beta1_adap_weight*g_bar_adaptive+(1-beta1_adap_weight)*g_adaptive;
    v_bar_adaptive = beta2_adap_weight*v_bar_adaptive+(1-beta2_adap_weight)*v_adaptive;
    
    % set stepsize
    if iter>=tau_threshold
        stepsize = eps0*tau_threshold/iter;
    else
        stepsize = eps0;
    end
    
    % update lambda
    lambda = lambda+stepsize*g_bar_adaptive./sqrt(v_bar_adaptive);
    
    LB(iter) = mean(h_lambda);
    
    if iter>=t_w
        LB_bar(iter-t_w+1) = mean(LB(iter-t_w+1:iter));  % calculate the smoothed LB
        disp(['    iteration ',num2str(iter),'|| smooth LB: ',num2str(LB_bar(iter-t_w+1), '%0.4f')]);
    else
        disp(['    iteration ',num2str(iter),'|| LB: ',num2str(LB(iter), '%0.4f')]);
    end
       
    if (iter>t_w)&&(LB_bar(iter-t_w+1)>=max(LB_bar))
        lambda_best = lambda;
        patience = 0;
    else
        patience = patience+1;
    end
    
    if (patience>patience_max)||(iter>max_iter) stop = true; end 
    
    iter = iter+1;
 
end
lambda = lambda_best;
mu_VBSL = lambda(1:d);
L = vechinv(lambda(d+1:end),2); Sigma_inv = L*L'; 
Sigma_VBSL = eye(d)/Sigma_inv;

% plot smoothed lower bound
figure(1)
plot(LB_bar)
title('Smoothed Lower Bound')

% Generate samples for theta (in the original constrained space) and plot
% the posterior density 
M = 1000;
theta_tilde = mvnrnd(mu_VBSL,Sigma_VBSL,M);
alpha_tilde = theta_tilde(:,1); 
beta_tilde = theta_tilde(:,2);
gamma_tilde = theta_tilde(:,3); 
delta = theta_tilde(:,4);
alpha = (2*exp(alpha_tilde)+1.1)./(1+exp(alpha_tilde));
beta = (exp(beta_tilde)-1)./(exp(beta_tilde)+1);
gamma = exp(gamma_tilde);
theta_samples = [alpha,beta,gamma,delta];
posterior_mean_estimate = mean(theta_samples) % estimate of posterior mean

figure(2)
fontsize = 15;
x_plot = mean(alpha)-4*std(alpha):0.001:mean(alpha)+4*std(alpha);
y_plot = ksdensity(alpha,x_plot);
subplot(1,4,1)
plot(x_plot,y_plot,'-','LineWidth',2);
xlabel('\alpha','FontSize', fontsize)
title('posterior density of \alpha')

x_plot = mean(beta)-4*std(beta):0.001:mean(beta)+4*std(beta);
y_plot = ksdensity(beta,x_plot);
subplot(1,4,2)
plot(x_plot,y_plot,'-','LineWidth',2);
xlabel('\beta','FontSize', fontsize)
title('posterior density of \beta')

x_plot = mean(gamma)-4*std(gamma):0.001:mean(gamma)+4*std(gamma);
y_plot = ksdensity(gamma,x_plot);
subplot(1,4,3)
plot(x_plot,y_plot,'-','LineWidth',2);
xlabel('\gamma','FontSize', fontsize)
title('posterior density of \gamma')

x_plot = mean(delta)-4*std(delta):0.001:mean(delta)+4*std(delta);
y_plot = ksdensity(delta,x_plot);
subplot(1,4,4)
plot(x_plot,y_plot,'-','LineWidth',2);
xlabel('\delta','FontSize', fontsize)
title('posterior density of \delta')



