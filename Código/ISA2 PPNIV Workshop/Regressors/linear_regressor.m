%%Train regressor - Linear
clear all
close all
%hist_folder = 'histograms_full_set_3';
hist_folder = 'histograms_full_set_3_Highway';

%load ('../ISA2_v1/Highway/speed_H1.mat')
%load ('../ISA2_v1/Highway/speed_H2.mat')
load ('../../Swiftnet/configs/out/ISA2/Highway/speed_H1.mat');
load ('../../Swiftnet/configs/out/ISA2/Highway/speed_H2.mat');
%load ('../../Swiftnet/configs/out/ISA2/Urban/speed_U2.mat');

%hist_train = load(fullfile(hist_folder, 'histogramsautovia2.mat'));
%hist_test = load(fullfile(hist_folder, 'histogramsautovia.mat'));
hist_train = load(fullfile('../../Swiftnet/configs/out/Results/' ,hist_folder, 'histogramsH2.mat'));
hist_test = load(fullfile('../../Swiftnet/configs/out/Results/', hist_folder, 'histogramsH1.mat'));
X_train = [hist_train.hist]';
X_test = [hist_test.hist]';
Y_test = [speed_a1(1:end)]';
Y_train = [speed_a2(1:end)]';

Mdl = fitlm(X_train, Y_train);
YHat = predict(Mdl, X_test);
YHat(YHat<0) = 0;

figure(1) 
plot(Y_test)
hold on
%B= 1/15*ones(15,1);
B= 1/20*ones(20,1);
speed_pred = filter(B,1,YHat);
plot(speed_pred)
error = mean(abs(Y_test(1:end) - speed_pred(1:end)))
lg = legend('Proper Speed', 'Proper Estimated Speed')
lg.FontSize = 10;
title('Linear Regressor')
xlabel('Images')
ylabel('Speed')
