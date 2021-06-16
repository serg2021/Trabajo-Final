%%Train regressor - SVM
clear all
close all
%hist_folder = 'histograms_full_set_3';
hist_folder = 'histograms_full_set_3_Highway';

%load ('../ISA2_v1/Highway/speed_H1.mat')
%load ('../ISA2_v1/Highway/speed_H2.mat')
load ('../../Swiftnet/configs/out/ISA2/Highway/speed_H1.mat');
load ('../../Swiftnet/configs/out/ISA2/Highway/speed_H2.mat');

%hist_train = load(fullfile(hist_folder, 'histogramsautovia2.mat'));
%hist_test = load(fullfile(hist_folder, 'histogramsautovia.mat'));
hist_train = load(fullfile('../../Swiftnet/configs/out/Results/' ,hist_folder, 'histogramsH2.mat'));
hist_test = load(fullfile('../../Swiftnet/configs/out/Results/', hist_folder, 'histogramsH1.mat'));
X_train = hist_train.hist';
X_test = hist_test.hist';
Y_train = speed_a2';
Y_test = speed_a1';

Mdl = fitrsvm(X_train,Y_train,'KernelFunction', 'linear', 'KernelScale', 'auto');
YHat = predict(Mdl,X_test);
YHat(YHat<0)=0;

figure(1) 
plot(Y_test(1:end))
hold on
B= 1/10*ones(10,1);
speed_pred = filter (B,1,YHat)
plot(speed_pred(1:end))
error = mean(abs(Y_test(1:end) - speed_pred(1:end)))
lg = legend('Proper Speed', 'Proper Estimated Speed')
lg.FontSize = 10;
title('SVR')
xlabel('Images')
ylabel('Speed')

