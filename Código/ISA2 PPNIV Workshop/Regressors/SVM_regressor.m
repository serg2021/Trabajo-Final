%%Train regressor - SVM
clear all
close all

%hist_folder = 'histograms_full_set_4';
%hist_folder = 'histograms_full_set_1_Highway';
hist_folder = 'histograms_full_set_1_Urban';

%load ('../ISA2_v1/Highway/speed_H1.mat')
%load ('../ISA2_v1/Highway/speed_H2.mat')
%speed_h1 = load ('../../Swiftnet/configs/out/ISA2/Highway/speed_H1.mat');
%speed_h2 = load ('../../Swiftnet/configs/out/ISA2/Highway/speed_H2.mat');
speed_u1 = load ('../../Swiftnet/configs/out/ISA2/Urban/speed_U1.mat');
speed_u2 = load ('../../Swiftnet/configs/out/ISA2/Urban/speed_U2.mat');
speed_u3 = load ('../../Swiftnet/configs/out/ISA2/Urban/speed_U3.mat');

%hist_train = load(fullfile(hist_folder, 'histogramsautovia2.mat'));
%hist_test = load(fullfile(hist_folder, 'histogramsautovia.mat'));
%hist_train = load(fullfile('../../Swiftnet/configs/out/Results/', hist_folder, 'histogramsH2.mat'));
%hist_test = load(fullfile('../../Swiftnet/configs/out/Results/', hist_folder, 'histogramsH1.mat'));
hist_train = [load(fullfile('../../Swiftnet/configs/out/Results/' , hist_folder, 'histogramsU1.mat')),load(fullfile('../../Swiftnet/configs/out/Results/' ,hist_folder, 'histogramsU2.mat'))];
hist_test = load(fullfile('../../Swiftnet/configs/out/Results/', hist_folder, 'histogramsU3.mat'));
X_train = [hist_train.hist]';
X_test = hist_test.hist';
%Y_train = [speed_h2.speed_a2(1:end)]';
%Y_test = [speed_h1.speed_a1(1:end)]';
Y_train = [speed_u1.speed_2fps(1:end), speed_u2.speed_2fps(1:end)];
Y_test = [speed_u3.speed_2fps(1:end)]';

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
ylabel('Speed');

img = imread('../../Swiftnet/datasets/ISA2/Highway/H1/0001541.jpeg');
img2 = imresize(img, 0.75);
b = string(Y_test(1192,1));
c = string(speed_pred(1192,1));
text_1 = ['Velocidad real: ' + b + ' km/h'];
text_2 = ['Velocidad estimada: ' + c + ' km/h'];
position = [1 1];
position2 = [1 24];
box_color = {'white'};
box_color2 = {'green'};
RGB = insertText(img2, position, text_1,'BoxOpacity', 1, 'BoxColor', box_color, 'TextColor', 'black');
RGB2 = insertText(RGB, position2, text_2, 'BoxOpacity', 1, 'BoxColor', box_color2, 'TextColor', 'black');
figure(2)
imshow(RGB2);
