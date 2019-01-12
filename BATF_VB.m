function [model] = BATF_VB(dense_tensor,sparse_tensor,varargin)
% Bayesian Augmented Tensor Factorization (BATF) using Variational Inference.

% Retrieve parameters from input
dim = size(sparse_tensor);
d = length(dim);

ip = inputParser;
ip.addParameter('CP_rank',30,@isscalar);
ip.addParameter('maxiter',200,@isscalar);
ip.parse(varargin{:});

r = ip.Results.CP_rank;
maxiter = ip.Results.maxiter;

% Initialization
pos_obs = find(sparse_tensor~=0);
pos_tst = find(dense_tensor>0 & sparse_tensor==0);
binary_tensor = zeros(dim);
binary_tensor(pos_obs) = 1;
nObs = length(pos_obs);

beta0 = 1;
nu0 = r;
m0 = zeros(r,1);
W0 = eye(r);
tau_epsilon = 1;
mu0 = 0;
tau0 = 1;
a0 = 1e-6;
b0 = 1e-6;

mu_glb = 0;
v = cell(d,1);
U = cell(d,1);
Sigma_U = cell(d,1);
for k = 1:d
    v{k} = 5*randn(dim(k),1);
    U{k} = 1.5*randn(dim(k),r);
    Sigma_U{k} = repmat(eye(r), [1 1 dim(k)]);
end
bias_tensor = vec_combination(v);
factor_tensor = cp_combination(U,dim);

EUUT = cell(d,1);
for k = 1:d
    EUUT{k} = (reshape(Sigma_U{k}, [r*r, dim(k)]))';
end

tau_bias = cell(d,1);
nu_U = cell(d,1);
beta_U = cell(d,1);
W_U = cell(d,1);
Lambda_U = cell(d,1);
mu_U = cell(d,1);

LB = 0;

%% VB for BATF model.
rmse = zeros(maxiter,1);
mape = zeros(maxiter,1);
fprintf('\n------Bayesian Augmented Tensor Factorization using Variational Inference------\n');

% Visualize the results
scrnsz = get(0,'ScreenSize');
h = figure('Position',[scrnsz(3)*0.15 scrnsz(4)*0.15 scrnsz(3)*0.7 scrnsz(4)*0.7]);

for iter = 1:maxiter
    % Update global \mu
    tau_mu = tau_epsilon*nObs+tau0;
    Ez = sparse_tensor - bias_tensor - factor_tensor;
    mu_glb(iter) = tau_mu^(-1)*(tau_epsilon*sum(Ez(pos_obs)) + tau0*mu0);
    
    for k = 1:d
        % Update bias v^{(k)}
        vslashk = v;
        vslashk{k} = zeros(dim(k),1);
        bias_tensor = vec_combination(vslashk);
        Ef = binary_tensor.*(sparse_tensor - mu_glb(iter) - bias_tensor - factor_tensor);
        tau_bias{k} = tau_epsilon*sum(ten2mat(binary_tensor, dim, k),2) + tau0;
        v{k} = tau_bias{k}.^(-1).*(tau_epsilon*sum(ten2mat(Ef, dim ,k),2) + tau0*mu0);
    end
    bias_tensor = vec_combination(v);

    for k = 1:d
        % Update hyper-parameters \Lambda^{(k)} and \mu^{(k)}.
        U_bar = mean(U{k},1)';
        nu_U{k} = dim(k)+nu0;
        beta_U{k} = dim(k)+beta0;
        W_U{k} = (inv(W0)+(dim(k)-1)*cov(U{k})+dim(k)*beta0/beta_U{k}*(U_bar-m0)*(U_bar-m0)')^(-1);
        Lambda_U{k} = nu_U{k}*W_U{k};
        mu_U{k} = (dim(k)*U_bar+beta0*m0)./beta_U{k};
        
        % Update factor matrice U^{(k)}.
        EkUUT = reshape(khatrirao_fast(EUUT{[1:k-1,k+1:d]},'r')' * ten2mat(binary_tensor,dim,k)',[r,r,dim(k)]);
        Ew = ten2mat(binary_tensor.*(sparse_tensor - mu_glb(iter) - bias_tensor),dim,k)';
        EkUTY = tau_epsilon*khatrirao_fast(U{[1:k-1,k+1:d]},'r')'*Ew + ones(r,dim(k)).*(Lambda_U{k}*mu_U{k});
        for i = 1:dim(k)
            Sigma_U{k}(:,:,i) = (tau_epsilon * EkUUT(:,:,i) + Lambda_U{k})^(-1);
            U{k}(i,:) = (Sigma_U{k}(:,:,i) * EkUTY(:,i))';
        end
        EUUT{k} = (reshape(Sigma_U{k}, [r*r, dim(k)]) + khatrirao_fast(U{k}',U{k}'))';
    end
    
    % Compute the estimated tensor.
    factor_tensor = cp_combination(U,dim);
    tensor_hat = mu_glb(iter) + bias_tensor + factor_tensor;
    mape(iter,1) = sum(abs(dense_tensor(pos_tst)-tensor_hat(pos_tst))./dense_tensor(pos_tst))./length(pos_tst);
    rmse(iter,1) = sqrt(sum((dense_tensor(pos_tst)-tensor_hat(pos_tst)).^2)./length(pos_tst));
    
    % Update precision \tau_{\epsilon}.
    Emuglb2 = nObs*mu_glb(iter)^2;
    Ebias2 = bias_tensor(pos_obs)'*bias_tensor(pos_obs);
    % Efactor2 = binary_tensor(:)' * khatrirao_fast(EUUT,'r') * ones(r*r,1);
    Efactor2 = 0;
    temp0 = cell(d,1);
    for j = 1:r
       for k = 1:d
           temp0{k} = EUUT{k}(:,(j-1)*r+1:j*r);
       end
       Efactor2 = Efactor2 + binary_tensor(:)'*khatrirao_fast(temp0,'r')*ones(r,1);
    end
    Ecomb = sum(mu_glb(iter)*bias_tensor(pos_obs))+sum(mu_glb(iter)*factor_tensor(pos_obs))+bias_tensor(pos_obs)'*factor_tensor(pos_obs);
    EYstar2 = Emuglb2 + Ebias2 + Efactor2 + 2*Ecomb;
    Eerr = sparse_tensor(:)'*sparse_tensor(:) - 2*sparse_tensor(:)'*tensor_hat(:) + EYstar2;
    a_tau = a0+0.5*nObs;
    b_tau = b0+0.5*Eerr;
    tau_epsilon = a_tau/b_tau;
    
    % Evaluate lower bound
    % Y | \mu_glb, v, U, \tau_epsilon
    EY = -0.5*nObs*safelog(2*pi)+0.5*nObs*(psi(a_tau)-safelog(b_tau))-0.5*tau_epsilon*Eerr;
    % \mu_glb | \mu0, \tau0
    Emuglb = -0.5*safelog(2*pi)+0.5*safelog(tau0)-0.5*tau0*(mu_glb(iter)^2+tau_mu^(-1)-2*mu_glb(iter)*mu0+mu0^2);
    Ebias = 0; % v | \mu0, \tau0
    for k = 1:d
        temp1 = -0.5*dim(k)*safelog(2*pi);
        temp2 = 0.5*dim(k)*safelog(tau0);
        temp3 = -0.5*tau0*(sum(v{k}.^2)+sum(tau_bias{k}.^(-1))-2*sum(mu0*v{k})+dim(k)*mu0^2);
        Ebias = Ebias + temp1 + temp2 + temp3;
    end
    EU = 0; % U | \mu_U, \Lambda_U
    for k = 1:d
        temp1 = -0.5*dim(k)*r*safelog(2*pi);
        temp2 = 0.5*dim(k)*(sum(psi(0.5*(nu_U{k}+1-1:r)))+r*safelog(2)+safelog(det(W_U{k})));
        temp3 = 0;
        for i = 1:dim(k)
           temp3 = temp3 - 0.5*( (U{k}(i,:)'-mu_U{k})'*Lambda_U{k}*(U{k}(i,:)'-mu_U{k}) + trace(Lambda_U{k}*(Sigma_U{k}(:,:,i)+(beta_U{k}*Lambda_U{k})^(-1))) ); 
        end
        EU = EU + temp1 + temp2 + temp3; 
    end
    EmuLambda = 0; % \mu_U, \Lambda_U | \mu0_U, \beta0, \nu0, \W0
    for k = 1:d
        temp1 = -0.5*r*safelog(2*pi);
        temp2 = 0.5*(r*safelog(beta0)+sum(psi(0.5*(nu_U{k}+1-1:r)))+r*safelog(2)+safelog(det(W_U{k})));
        temp3 = -0.5*beta0*(r*beta_U{k}+(mu_U{k}-m0)'*Lambda_U{k}*(mu_U{k}-m0));
        temp4 = -0.5*nu0*r*safelog(2)-0.5*nu0*safelog(det(W0))-0.25*r*(r-1)*safelog(pi)-sum(safelog(gamma(0.5*(nu0+1-1:r))));
        temp5 = 0.5*(nu0-r-1)*(sum(psi(0.5*(nu_U{k}+1-1:r)))+r*safelog(2)+safelog(det(W_U{k})));
        temp6 = -0.5*trace(W0^(-1)*Lambda_U{k});
        EmuLambda = EmuLambda + temp1 + temp2 + temp3 + temp4 + temp5 + temp6; 
    end
    % \tau_epsilon | \a0, \b0
    Etauepsilon = -safelog(gamma(a0))+a0*safelog(b0)+(a0-1)*(psi(a_tau)-safelog(b_tau))-b0*tau_epsilon;
    Smuglb = 0.5*(1+safelog(2*pi*tau_mu^(-1)));
    Sbias = 0;
    for k = 1:d
        Sbias = Sbias + 0.5*sum(1+safelog(2*pi*tau_bias{k}.^(-1)));
    end
    SU = 0;
    for k = 1:d
        for i = 1:dim(k)
            SU = SU + 0.5*(r+safelog(det(2*pi*Sigma_U{k}(:,:,i)))); 
        end
    end
    SmuLambda = 0;
    for k = 1:d
        temp1 = 0.5*(r+safelog(det(2*pi*(beta_U{k}*Lambda_U{k})^(-1))));
        temp2 = 0.5*nu_U{k}*safelog(det(W_U{k}))+0.5*nu_U{k}*r*safelog(2)+0.25*r*(r-1)*safelog(pi)+sum(safelog(gamma(0.5*(nu_U{k}+1-1:r))));
        temp3 = -0.5*(nu_U{k}-r-1)*(sum(psi(0.5*(nu_U{k}+1-1:r)))+r*safelog(2)+safelog(det(W_U{k})))+0.5*nu_U{k}*r;
        SmuLambda = SmuLambda + temp1 + temp2 + temp3; 
    end
    Stauepsilon = safelog(gamma(a_tau))-(a_tau-1)*psi(a_tau)-safelog(b_tau)+a_tau;
    LB(iter) = EY+Emuglb+Ebias+EU+EmuLambda+Etauepsilon+Smuglb+Sbias+SU+SmuLambda+Stauepsilon;
    
    % Print the results.
    fprintf('Epoch = %g, MAPE = %g, RMSE = %g km/h .\n',iter,mape(iter),rmse(iter));
    set(0,'CurrentFigure',h);
    for k = 1:d
        subplot(2,d,k);imagesc(U{k});colormap parula;colorbar;
    end
    subplot(2,d,4); yyaxis left; plot(1:iter,mape(1:iter)); title('Test results'); xlabel('Epoch'); ylabel('MAPE (ratio)'); ylim([0,0.25]); 
    yyaxis right; plot(1:iter,rmse(1:iter)); ylabel('RMSE (km/h)'); ylim([3,5]); legend('MAPE','RMSE'); grid on;
    subplot(2,d,5); plot(LB, '-r.','LineWidth',1.5,'MarkerSize',10 ); title('Lower bound'); xlabel('Epoch'); grid on;
    subplot(2,d,6); plot(tau_epsilon-0.015:0.005:tau_epsilon+0.015, gampdf(tau_epsilon-0.015:0.005:tau_epsilon+0.015, a_tau, 1./b_tau), 'r-'); title('Posterior pdf'); xlabel('Noise precision \tau_{\epsilon}'); grid on;
    set(findall(h,'type','text'),'fontSize',12);
    drawnow;
end

finalResults = cell(2,1);
finalResults{1} = mape;
finalResults{2} = rmse;

%% Output
model.tensorHat = tensor_hat;
model.mu = mu_glb;
model.bias = v;
model.biasTensor = bias_tensor;
model.factorMatrices = U;
model.lowerBound = LB;
model.finalResults = finalResults;

end