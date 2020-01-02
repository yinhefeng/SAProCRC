clear
clc
close all

addpath('ompbox10')

% load data and the indices of training samples for 10 runs
load('randomfaces4ar.mat');
load('Tr_ind_AR.mat')

experiments = size(Tr_ind,1); %10 repeated experiments
acc = zeros(1,experiments); %record the accuracy

ClassNum = length(unique(gnd));%number of classes

% parameters
params.model_type = 'ProCRC';
params.gamma = 1e-3;
params.lambda = 1e-2;
sparsity = 50;

for ii=1:experiments
    ii
    % indices of training and test samples
    train_ind = logical(Tr_ind(ii,:));
    test_ind = ~train_ind;
    
    % data and label vectors of training and test samples
    training_feats = fea(:,train_ind);
    testing_feats = fea(:,test_ind);
    train_label = gnd(:,train_ind);
    test_label = gnd(:,test_ind);
    
    % label matrix of training samples
    H_train = full(ind2vec(train_label,ClassNum));
    
    % normalize to unit L2 norm
    train = normc(training_feats);
    Y = normc(testing_feats);
    
    Phi = train;
    
    % parameters for ProCRC
    fr_dat_split = [];
    fr_dat_split.tr_descr = train;
    fr_dat_split.tr_label = vec2ind(H_train);
    fr_dat_split.tt_descr = testing_feats;
    fr_dat_split.tt_label = test_label;
    
    params.class_num = size(H_train,1);
    
    % coefficient matrix of ProCRC
    A_check = ProCRC(fr_dat_split, params);
    
    % coefficient matrix of SRC
    G = Phi'*Phi;
    A_hat = omp(Phi'*Y,G,sparsity);
    
    % augmented coefficient
    A_aug = normc(A_check + A_hat);
    
    % classification
    Score = H_train * A_aug;
    [~,pre_label] = max(Score);
    acc(ii) = sum(pre_label==test_label)/length(test_label)*100
end

% mean and standard deviation of 10 experiments
mean(acc)
std(acc)