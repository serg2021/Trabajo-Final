clear all
close all

%status = system('../Swiftnet/python eval.py configs/rn18_single_scale.py')
[status, cmdout] = system('source /home/sergios/miniconda2/etc/profile.d/conda.sh && conda activate swiftnet4 && python ../Swiftnet/eval.py ../Swiftnet/configs/rn18_single_scale.py &');

pause('on');
pause(10);
hist_folder = 'histograms_full_set_1_Urban';
speed_u1 = load ('../Swiftnet/configs/out/ISA2/Urban/speed_U1.mat');
speed_u2 = load ('../Swiftnet/configs/out/ISA2/Urban/speed_U2.mat');
speed_u3 = load ('../Swiftnet/configs/out/ISA2/Urban/speed_U3.mat');
IMG_ROOT =['../Swiftnet/configs/out/ISA2/'];
cur_path = fullfile(IMG_ROOT);
images_names = dir(cur_path);
results = dir(fullfile(cur_path, '*.mat'));
jj = 1;
tt = 1;
while(numel(results) > 0)  
    fn = results(jj).name(1:end-11);
    tmp = load(fullfile(cur_path, results(jj).name));      
    raw_result = tmp.data;
    %raw_result = permute(raw_result, [1 2 3]);
    %[~, result] = max(raw_result, [], 3);           
    result = uint8(raw_result(1:1024, 1:2048));
            
    %Get histograms
    histog = ObtainHistograms(result, 1);
    
    hist_train = [load(fullfile('../Swiftnet/configs/out/Results/' , hist_folder, 'histogramsU1.mat')),load(fullfile('../Swiftnet/configs/out/Results/' ,hist_folder, 'histogramsU2.mat'))];
    hist_test = histog;
    
    X_train = [hist_train.hist]';
    X_test = hist_test;
    
    Y_train = [speed_u1.speed_2fps(1:end), speed_u2.speed_2fps(1:end)];
    Y_test = [speed_u3.speed_2fps(1:end)]';
    
    Mdl = fitrsvm(X_train,Y_train,'KernelFunction', 'linear', 'KernelScale', 'auto');
    YHat = predict(Mdl,X_test);
    YHat(YHat<0)=0;
    
    speed_pred(tt,:) = YHat;
    %B= 1/10*ones(10,1);
    %speed_pred = filter (B,1,YHat);
    %error = mean(abs(Y_test(1:end) - speed_pred(1:end)));
    
    img = imread(strcat('../Swiftnet/datasets/ISA2/Urban/U3/', tmp.name ,'.jpeg'));
    img2 = imresize(img, 0.75);
    b = string(Y_test(tt,1));
    c = string(YHat);
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
    tt = tt + 1;
    delete(fullfile(cur_path, results(jj).name));
    results = dir(fullfile(cur_path, '*.mat'));
    
end

error = mean(abs(Y_test(1:end) - speed_pred(1:end)));

