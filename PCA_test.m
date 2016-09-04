close all
%--

% load('/home/michele/Desktop/fra1.mat')
% imgs = images;

%--

load('/home/michele/Development/IDRA/IDRA/data/imgData.mat')

imgData = imgData';
imgData = reshape(imgData, size(imgData,1), 120, 160);
imgs = imgData;

%--

imgs = double(imgs);



samples = randsample(size(imgs, 1), 1000);
imgs = imgs(samples, :,:);


ch = ClassificationHelper();
ih = ImageHelper();

in_size = size(imgs);
img_size = [in_size(end-1), in_size(end)];

n_patch = 200;
patch_base_size = 12;

[norm_patches, ~] = ih.GeneratePatches(imgs, n_patch, patch_base_size);
pc_raw = ch.DoPCA(norm_patches, 1);
[ics, A, W] = fastica(pc_raw.score', 'numOfIc', patch_base_size^2);

backup = norm_patches;
%--


norm_patches = backup;

norm_patches = zscore(norm_patches')';

samples = randsample(size(norm_patches, 1), n_patch * size(imgs, 1));


pc_raw = ch.DoPCA(norm_patches, 1);

sigin = pc_raw.score(samples, :)';

nics = 256;

[ics, A, W] = fastica(sigin, 'numOfIc', nics);
nics = size(ics,1);

%--
% samples = randsample(size(norm_patches, 1), 100000);
% 
% tt = (W(1:2,:) * pc_raw.score(samples,:)')' ;
% 
% [ids, centroids, sumd, dist] = kmeans(tt, 64, 'EmptyAction','singleton', 'Distance', 'cosine', 'MaxIter', 500);
% pc_ics = ch.DoPCA(tt, 1);
% ch.GScatter3(tt, ids, 2, 'ics = W * sigin');

%--

rec_A = A' * pc_raw.coeff';

sig_rec = sigin';
sig_rec = sig_rec(1:200, :)*pc_raw.coeff';


rec_sigin = (A(:, 1:nics)*ics(1:nics, :))'*pc_raw.coeff';

plotted_patches = rec_A;

n_patches = size(plotted_patches, 1);

ncols = 20;
nrows = floor(n_patches / ncols) + 1;

figure(9);
colormap gray;
for ii = 1:nrows
    for jj = 1:ncols
        ind = (ii-1) * ncols + jj;
        
        if ind > n_patches
            break;
        end
        
        subplot(nrows, ncols, ind);
        imagesc(reshape(plotted_patches(ind,:), patch_base_size, patch_base_size));
        set(gca,'XTickLabel',{});
        set(gca,'YTickLabel',{});
    end
end

    