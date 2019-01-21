clc;
clear;

%% Model Training
low_rank = 10;
max_iteration = 200;
mape = zeros(3,1);
rmse = zeros(3,1);
for rate = [1 3 5]
    ratio = 0.1*rate;
    load tensor;
    [dense_tensor,sparse_tensor] = ms_scenario(tensor,'ms','fiber','missing_rate',ratio);
    [model] = BATF_VB(dense_tensor,sparse_tensor,'CP_rank',low_rank,'maxiter',max_iteration);
    mape(rate,1) = model.finalResults{1}(max_iteration,1);
    rmse(rate,1) = model.finalResults{2}(max_iteration,1);
end

%% Results Plotting
dim = size(tensor);

% Road #1
road = 1;

mu = model.mu(end);
bias_1 = reshape(reshape(model.biasTensor(road,:,:),dim(2),dim(3))',1,[])';
mu_bias_1 = mu + bias_1;
tensorHat_1 = reshape(reshape(model.tensorHat(road,:,:),dim(2),dim(3))',1,[])';

dense_tensor_1 = reshape(reshape(dense_tensor(road,:,:),dim(2),dim(3))',1,[])';
sparse_tensor_1 = reshape(reshape(sparse_tensor(road,:,:),dim(2),dim(3))',1,[])';

output = table(dense_tensor_1,sparse_tensor_1,mu_bias_1,tensorHat_1);
writetable(output,'road1_fiber_ms50_r10.csv');
