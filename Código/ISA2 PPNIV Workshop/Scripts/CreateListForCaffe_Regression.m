%Example of List for Caffe: urban (we use seq1 and seq3 for training and seq2 for testing

%Reading the images
imagesTrain = dir('../ISA2_v1/Highway/H2');
imagesTrain = imagesTrain(3:end);
imagesTest = dir('../ISA2_v1/Highway/H1');
imagesTest = imagesTest(3:end);

%Reading the speed
load('../ISA2_v1/Highway/speed_H1.mat')
load('../ISA2_v1/Highway/speed_H2.mat')

%Creating .txt files
fid = fopen('train.txt', 'w');
fid2 = fopen('train_id.txt', 'w');            
fid3 = fopen('val.txt', 'w');       
fid4 = fopen('val_id.txt', 'w');             



%Writing .txt files            
for i=1:size(imagesTrain)
    fprintf(fid, '%s %d\n', ['H1/' imagesTrain(i).name], round(speed_a2(1,i)));
    fprintf(fid2, '%s\n', imagesTrain(i).name(1:end-5)); 
end

for i=1:size(imagesTest)
    fprintf(fid3, '%s %d\n', ['H2/' imagesTest(i).name],round(speed_a1(1,i))); 
    fprintf(fid4, '%s\n', imagesTest(i).name(1:end-5)); 
end
