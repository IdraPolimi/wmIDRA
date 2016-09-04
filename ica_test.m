% 
% imagefiles = dir('*.png');      
% nfiles = length(imagefiles);    % Number of files found
% 
% for ii=1:nfiles
%    currentfilename = imagefiles(ii).name;
%    currentimage = imread(currentfilename);
%    images(ii,:,:) = currentimage;
% end

ch = ClassificationHelper();


load('/home/michele/Desktop/fra1.mat')
% load('/home/michele/Desktop/imgs/faces/faces.mat');
% images = imgs;
img_h = size(images, 2);
img_w = size(images, 3);

images = reshape(images, size(images, 1), img_h * img_w);

imgs = double(images);

pc_sigin = ch.DoPCA(imgs, 1);
in_pca = pc_sigin.score;

pc_img = in_pca * pc_sigin.coeff';

n_ics = 32;


[ics, A, W] = fastica(pc_sigin.score, 'numOfIc', n_ics);
inv_ics = pinv(ics);
n_ics = size(ics,1);


ica_rec = (A*ics)*pc_sigin.coeff';



mod = repmat(sqrt( sum(in_pca.^2, 2) ), 1, size(inv_ics, 2)) .* ...
repmat(sqrt( sum(inv_ics.^2, 1) ), size(in_pca, 1), 1);
            
ica_weights = (in_pca*inv_ics) ./mod;

sim = sqrt( sum(ica_weights.^2, 2) );

pc_ica = ch.DoPCA(ica_weights', 1);

ch.GScatter3(pc_ica.score, ones(size(pc_ica.coeff', 1), 1), 15, 'ics');

