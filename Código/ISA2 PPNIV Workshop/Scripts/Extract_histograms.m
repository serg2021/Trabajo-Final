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
divisions = 1;
% Folder to save the histograms
hist_folder = 'histograms_full_set_1_Urban';    

% Directories
%ROOT_PATH = '../DeepLab/cityscapes/';
ROOT_PATH = '../../Swiftnet/configs/out/';
%IMG_ROOT =['../ISA2_v1/Highway/H2'];
IMG_ROOT = '../../Swiftnet/configs/out/ISA2/Urban/U3';
%SAVE_HIST_PATH = ['./'];
SAVE_HIST_PATH = '../../Swiftnet/configs/out/Results';

%pepe = load(fullfile('../../Swiftnet/configs/out/Results/histograms_full_set_1_Urban/histogramsU3.mat'));
%pepe2 = load(fullfile('../../ISA2 PPNIV Workshop/Regressors/histograms_full_set_1_urban/histogramsseq2.mat'));

for kk = 1
    cur_path = fullfile(IMG_ROOT);
    images_names = dir(cur_path);

    %vis_save_folder = fullfile('vis_res', load_folder);
    %save_folder = fullfile('res', load_folder);


    results = dir(fullfile(cur_path, '*.mat'));

    save_path = fullfile(ROOT_PATH, 'Grey');
    vis_save_path = fullfile(ROOT_PATH, 'Color');

    if ~exist(vis_save_path, 'dir')
    	mkdir(vis_save_path)
    end

    if ~exist(save_path, 'dir')
    	mkdir(save_path)
    end
        
    %Process .mat results
    for jj = 1 : numel(results)
    	fn = results(jj).name(1:end-11);

        tmp = load(fullfile(cur_path, results(jj).name));
        raw_result = tmp.data;
        %raw_result = permute(raw_result, [1 2 3]);
        %[~, result] = max(raw_result, [], 3);           
        result = uint8(raw_result(1:1024, 1:2048));
            
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
        save(fullfile(SAVE_HIST_PATH, hist_folder,['histograms', 'U3', '.mat']), 'hist');
        clearvars hist;
    end
end

close all