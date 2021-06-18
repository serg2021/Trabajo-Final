%%Train regressor - Boosting trees
clear all
close all

hist_folder = 'histograms_full_set_3_urban';
%hist_folder = 'histograms_full_set_1_Highway';
%hist_folder = 'histograms_full_set_3_Urban';

%load ('../ISA2_v1/Highway/speed_H1.mat');
%load ('../ISA2_v1/Highway/speed_H2.mat');
%load ('../../Swiftnet/configs/out/ISA2/Highway/speed_H1.mat');
%load ('../../Swiftnet/configs/out/ISA2/Highway/speed_H2.mat');
speed_u1 = load ('../../Swiftnet/configs/out/ISA2/Urban/speed_U1.mat');
speed_u2 = load ('../../Swiftnet/configs/out/ISA2/Urban/speed_U2.mat');
speed_u3 = load ('../../Swiftnet/configs/out/ISA2/Urban/speed_U3.mat');

hist_train = [load(fullfile(hist_folder, 'histogramsseq1.mat')), load(fullfile(hist_folder, 'histogramsseq3.mat'))];
hist_test = load(fullfile(hist_folder, 'histogramsseq2.mat'));
%hist_train = [load(fullfile('../../Swiftnet/configs/out/Results/' ,hist_folder, 'histogramsU1.mat')),load(fullfile('../../Swiftnet/configs/out/Results/' ,hist_folder, 'histogramsU2.mat'))];
%hist_test = load(fullfile('../../Swiftnet/configs/out/Results/', hist_folder, 'histogramsU3.mat'));
X_train = [hist_train.hist]';
X_test = [hist_test.hist]';
Y_train = [speed_u1.speed_2fps(1:end), speed_u2.speed_2fps(1:end)];
Y_test = [speed_u3.speed_2fps(1:end)]';


%Model parameters
n = size(X_train,1);
m = floor(log2(n - 1));
learnRate = [0.1 0.25 0.5 1];
numLR = numel(learnRate)
maxNumSplits = 2.^(0:m);
numMNS = numel(maxNumSplits);
numTrees = 50;
Mdl = cell(numMNS,numLR);

%%Extraemos un modelo para cada learning rate y n�mero m�ximo de Splits, empleando para
%%cada modelo 5-Fold Cross-Valdation
for k = 1:numLR
    k
    for j = 1:numMNS
        t = templateTree('MaxNumSplits',maxNumSplits(j));
        Mdl{j,k} = fitrensemble(X_train,Y_train,'NumLearningCycles',numTrees,...
            'Learners',t,'KFold',5,'LearnRate',learnRate(k));
    end
end

kflAll = @(x)kfoldLoss(x,'Mode','cumulative');
errorCell = cellfun(kflAll,Mdl,'Uniform',false);
error = reshape(cell2mat(errorCell),[numTrees numel(maxNumSplits) numel(learnRate)]);

[minErr,minErrIdxLin] = min(error(:));
[idxNumTrees,idxMNS,idxLR] = ind2sub(size(error),minErrIdxLin);

fprintf('\nMin. MSE = %0.5f',minErr)
fprintf('\nOptimal Parameter Values:\nNum. Trees = %d',idxNumTrees);
fprintf('\nMaxNumSplits = %d\nLearning Rate = %0.2f\n',...
    maxNumSplits(idxMNS),learnRate(idxLR))

%Best Model
tFinal = templateTree('MaxNumSplits',maxNumSplits(idxMNS));
MdlFinal = fitrensemble(X_train,Y_train,'NumLearningCycles',idxNumTrees,...
    'Learners',tFinal,'LearnRate',learnRate(idxLR))

YHat = predict(MdlFinal, X_test);

figure(1) 
plot(Y_test)
hold on
B= 1/15*ones(15,1);
speed_pred = filter(B,1,YHat)
plot(speed_pred)
error = mean(abs(Y_test(1:end) - speed_pred(1:end)))
lg = legend('Proper Speed', 'Proper Estimated Speed')
lg.FontSize = 10;
title('Boosting Trees')
xlabel('Images')
ylabel('Speed')


