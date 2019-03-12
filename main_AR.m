clear
clc
close all

%载入数据及对应的训练样本索引
load('randomfaces4ar.mat');
load('Tr_ind_AR.mat')

experiments = size(Tr_ind,1);%重复10次实验
acc = zeros(1,experiments);%保存每次的准确率

ClassNum = length(unique(gnd));%类别数
%参数设置
params.model_type = 'ProCRC';
params.gamma = 1e-3;
params.lambda = 1e-2;
sparsity = 50;

for ii=1:experiments
    ii
    %训练和测试样本索引
    train_ind = logical(Tr_ind(ii,:));
    test_ind = ~train_ind;
    
    %训练和测试样本对应的数据和标签
    training_feats = fea(:,train_ind);
    testing_feats = fea(:,test_ind);
    train_label = gnd(:,train_ind);
    test_label = gnd(:,test_ind);
    
    %训练样本标签矩阵
    H_train = full(ind2vec(train_label,ClassNum));
    
    %单位化
    train = normc(training_feats);
    Y = normc(testing_feats);
    
    Phi = train;
    
    %ProCRC参数设置
    fr_dat_split = [];
    fr_dat_split.tr_descr = train;
    fr_dat_split.tr_label = vec2ind(H_train);
    fr_dat_split.tt_descr = testing_feats;
    fr_dat_split.tt_label = test_label;
    
    params.class_num = size(H_train,1);
    
    %ProCRC得到的表示系数矩阵
    A_check = ProCRC(fr_dat_split, params);
    
    %SRC得到的表示系数矩阵
    G = Phi'*Phi;
    A_hat = omp(Phi'*Y,G,sparsity);
    
    %增强的表示系数
    A_aug = normc(A_check + A_hat);
    
    %分类并得到准确率
    Score = H_train * A_aug;
    [~,pre_label] = max(Score);
    acc(ii) = sum(pre_label==test_label)/length(test_label)*100
end
%10次实验准确率的均值和标准差
mean(acc)
std(acc)