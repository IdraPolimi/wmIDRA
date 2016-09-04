close all;
clear variables;
%    simpleclass_dataset   - Simple pattern recognition dataset.
%    cancer_dataset        - Breast cancer dataset.
%    crab_dataset          - Crab gender dataset.
%    glass_dataset         - Glass chemical dataset.
%    iris_dataset          - Iris flower dataset.
%    thyroid_dataset       - Thyroid function dataset.
%    wine_dataset          - Italian wines dataset.
ch = ClassificationHelper();

[data, targets] = ch.CreateDataset([.1 .4 .3 .2],1000, 900);
[data, targets] = thyroid_dataset;
data = data';
targets = targets';



% data = abalone_dataset;
% data = data';
% 
% targets = ch.GenerateTargets(data,1);
% data(:,1) = [];




[training_indeces, test_indeces, validation_indeces] = ch.SplitData(data, 0.6, 0.4);

training_data = data(training_indeces,:);
training_target = targets(training_indeces, :);


test_data = data(test_indeces,:);
test_target = targets(test_indeces, :);

%%


training_groups = zeros(length(training_target), 1);
for ii = 1:size(training_target,2)
    indeces = find(training_target(:,ii) == 1);
    training_groups(indeces) = ii;
end

test_groups = zeros(length(test_target), 1);
for ii = 1:size(test_target,2)
    indeces = find(test_target(:,ii) == 1);
    test_groups(indeces) = ii;
end
training_target = training_target';
test_target = test_target';


pc_train = ch.DoPCA(training_data, 1);
pc_test = ch.DoPCA(test_data, 1);

%%
close all;


ia = IAC(0, 0.2);

tset_length = length(training_data);

input = ia.NewFilterNode(size(training_data, 2), @NoFilter);

n_centroids = 32;
im1 = ia.NewIntentionalModule(input.index, tset_length, 2, n_centroids); 


while ia.IsBootstraping()
    input.SetInput(training_data);
    ia.Train();
end


%%

[train_sim, train_ica] = ch.ProcessIntentionalModule(ia, 1, 2, training_data);
[test_sim, test_ica] = ch.ProcessIntentionalModule(ia, 1, 2, test_data);


% ica_pc = ch.DoPCA(test_ica, 1);
% sim_pc = ch.DoPCA(test_sim, 1);
% 
% ch.GScatter3(ica_pc.score, test_groups, 5428, 'ICA');
% ch.GScatter3(sim_pc.score, test_groups, 5427, 'SIM');


%%



train_X = training_data';
net1 = trainSoftmaxLayer(train_X, training_target);
train_X = test_data';
train_Y = net1(train_X);

%--

pca_X = pc_train.score';
net2 = trainSoftmaxLayer(pca_X,training_target);
pca_X = pc_test.score';
pca_Y = net2(pca_X);
%--


sim_X = train_sim';
net2 = trainSoftmaxLayer(sim_X,training_target);
sim_X = test_sim';
sim_Y = net2(sim_X);

ica_X = train_ica';
net3 = trainSoftmaxLayer(ica_X,training_target);
ica_X = test_ica';
ica_Y = net3(ica_X);



figure(10005);
plotconfusion(  test_target, pca_Y, 'Softmax classifier', ...
                test_target, ica_Y,'Catogorization Module');

%%
    
% for ii = 1:length(test_data)
%      input.SetInput(test_data(ii,:));
% figure(999);
%      plot(test_data(ii,:));
%      ia.Update();
% end