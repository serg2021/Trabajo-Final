%Script para guardar histogramas a partir de la segmentación y generar las
%estructuras necesarias para la red de predicción
%
clear all; close all;

cityscapes_variables;

% To show the segmentes images
debug = 0;
% Save results for color visualization
do_save_results = 0;
% Save_histograms
save_histograms = 1;
% Number of divisions of the spatial pyramid
divisions = 4;
% Folder to save the histograms
hist_folder = 'histograms_full_set_4_urban';





load_folder = 'features';
net_id = 'resnet';
test_set = 'val';    
save_test_set = test_set(1:end-6);
feature_types = {'crf'};      

% Directories
ROOT_PATH = '../DeepLab/cityscapes/';
IMG_ROOT =['../ISA2_v1/Highway/H2']; 
SAVE_HIST_PATH = ['./'];




for kk = 1 : numel(feature_types)
    cur_path = fullfile(ROOT_PATH, load_folder, net_id, test_set, feature_types{kk})
    images_names = dir(cur_path);

    vis_save_folder = fullfile('vis_res', load_folder);
    save_folder = fullfile('res', load_folder);


    results = dir(fullfile(cur_path, '*.mat'));

    save_path = fullfile(ROOT_PATH,  save_folder, net_id, test_set, feature_types{kk});
    vis_save_path = fullfile(ROOT_PATH, vis_save_folder, net_id, test_set, feature_types{kk});

    if ~exist(vis_save_path, 'dir')
    	mkdir(vis_save_path)
    end

    if ~exist(save_path, 'dir')
    	mkdir(save_path)
    end
        
    %Process .mat results
    for jj = 1 : numel(results)
        jj
    	fn = results(jj).name(1:end-11);

        tmp = load(fullfile(cur_path, results(jj).name));
        raw_result = tmp.data;
        raw_result = permute(raw_result, [2 1 3]);
        [~, result] = max(raw_result, [], 3);           
        result = uint8(result(1:384, 1:640)) - 1;
            
        %Get histograms
        [histog] = ObtainHistograms(result, divisions);
        hist(:,jj) = histog;


        eval_result = TransformTrainidToId(result, cityscapes.trainID_to_id);

        %Show images debug
        if debug
        	figure(1)
            img = imread(fullfile(IMG_ROOT, [fn '.jpeg']));
            subplot(1, 2, 1), imshow(img)
            subplot(1, 2, 2), imshow(result, cityscapes.cmap_trainid);
            pause
        end
            
        %Save .png images
        if do_save_results
            imwrite(eval_result, fullfile(save_path, [fn '.png']));
            imwrite(result, cityscapes.cmap_trainid, fullfile(vis_save_path, [fn '.png']));
        end

	end
        
    %Save histograms
    if save_histograms
    	if ~exist(fullfile(SAVE_HIST_PATH, hist_folder), 'dir')
        	mkdir(fullfile(SAVE_HIST_PATH, hist_folder));
        end      
        save(fullfile(SAVE_HIST_PATH, hist_folder,['histograms', divisions, '.mat']), 'hist');
        clearvars hist;
    end
end

close all