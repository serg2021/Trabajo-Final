%Example of List for Caffe: urban (we use seq1 and seq3 for training and seq2 for testing

%Reading the images
images = dir('../ISA2_v1/Highway/H2');
images = images(3:end);


%Creating .txt files        
fid = fopen('val.txt', 'w');       
fid2 = fopen('val_id.txt', 'w');             


%Writing .txt files            
for i=1:size(images)
    fprintf(fid, '%s\n', ['H2/' images(i).name]);
    fprintf(fid2, '%s\n', images(i).name(1:end-5)); 
end
