%%Train regressor - Linear
clear all
close all

hist_folder = 'histograms_full_set_4';
%hist_folder = 'histograms_full_set_4_Highway';
%hist_folder = 'histograms_full_set_4_Urban';

%load ('../ISA2_v1/Highway/speed_H1.mat')
%load ('../ISA2_v1/Highway/speed_H2.mat')
speed_h1 = load ('../../Swiftnet/configs/out/ISA2/Highway/speed_H1.mat');
speed_h2 = load ('../../Swiftnet/configs/out/ISA2/Highway/speed_H2.mat');
%speed_u1 = load ('../../Swiftnet/configs/out/ISA2/Urban/speed_U1.mat');
%speed_u2 = load ('../../Swiftnet/configs/out/ISA2/Urban/speed_U2.mat');
%speed_u3 = load ('../../Swiftnet/configs/out/ISA2/Urban/speed_U3.mat');

%hist_train = [load(fullfile(hist_folder, 'histogramsseq1.mat')), load(fullfile(hist_folder, 'histogramsseq3.mat'))];
%hist_test = load(fullfile(hist_folder, 'histogramsseq2.mat'));
hist_train = load(fullfile(hist_folder, 'histogramsautovia2.mat'));
hist_test = load(fullfile(hist_folder, 'histogramsautovia.mat'));
X_train = [hist_train.hist]';
X_test = [hist_test.hist]';
Y_train = [speed_h2.speed_a2(1:end)]';
Y_test = [speed_h1.speed_a1(1:end)]';

Mdl = fitlm(X_train, Y_train);
YHat = predict(Mdl, X_test);
YHat(YHat<0) = 0;

figure(1) 
plot(Y_test)
hold on
%B= 1/15*ones(15,1);
B= 1/25000000*ones(25000000,1);
%B= 1/20*ones(20,1);
speed_pred = filter(B,1,YHat)
plot(speed_pred)
error = mean(abs(Y_test(1:end) - speed_pred(1:end)))
lg = legend('Proper Speed', 'Proper Estimated Speed')
lg.FontSize = 10;
title('Linear Regressor')
xlabel('Images')
ylabel('Speed')
