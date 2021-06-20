%%Train regressor - Lasso
clear all
close all

%hist_folder = 'histograms_full_set_3_urban';
%hist_folder = 'histograms_full_set_3_Highway';
hist_folder = 'histograms_full_set_3_Urban';
 
%load ('../ISA2_v1/Highway/speed_H1.mat')
%load ('../ISA2_v1/Highway/speed_H2.mat')
%load ('../../Swiftnet/configs/out/ISA2/Highway/speed_H1.mat');
%load ('../../Swiftnet/configs/out/ISA2/Highway/speed_H2.mat');
speed_u1 = load ('../../Swiftnet/configs/out/ISA2/Urban/speed_U1.mat');
speed_u2 = load ('../../Swiftnet/configs/out/ISA2/Urban/speed_U2.mat');
speed_u3 = load ('../../Swiftnet/configs/out/ISA2/Urban/speed_U3.mat');
 
%hist_train = [load(fullfile(hist_folder, 'histogramsseq1.mat')), load(fullfile(hist_folder, 'histogramsseq3.mat'))];
%hist_test = load(fullfile(hist_folder, 'histogramsseq2.mat'));
hist_train = [load(fullfile('../../Swiftnet/configs/out/Results/' ,hist_folder, 'histogramsU1.mat')),load(fullfile('../../Swiftnet/configs/out/Results/' ,hist_folder, 'histogramsU2.mat'))];
hist_test = load(fullfile('../../Swiftnet/configs/out/Results/', hist_folder, 'histogramsU3.mat'));
X_train = [hist_train.hist]';
X_test = hist_test.hist';
Y_train = [speed_u1.speed_2fps(1:end), speed_u2.speed_2fps(1:end)];
Y_test = [speed_u3.speed_2fps(1:end)]';


[B,FitInfo] = lasso(X_train, Y_train,'Alpha',1,'CV',10);
idxLambda1SE = FitInfo.Index1SE;
coef = B(:,idxLambda1SE);
coef0 = FitInfo.Intercept(idxLambda1SE);

YHat = X_test*coef + coef0;
YHat(YHat<0)=0;

figure(1)
plot(Y_test)
hold on
B= 1/15*ones(15,1);
speed_pred = filter (B,1,YHat);
plot(speed_pred)
error = mean(abs(Y_test(1:end) - speed_pred(1:end)));
lg = legend('Proper Speed', 'Proper Estimated Speed');
lg.FontSize = 10;
title('Lasso Regressor')
xlabel('Images')
ylabel('Speed')
