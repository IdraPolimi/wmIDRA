close all;
clear all;

load('/home/michele/Development/IDRA/IDRA/data/imgData.mat')

imgData = double(imgData');
img_h = 120;
img_w = 160;
imgData = reshape(imgData, size(imgData,1), img_h, img_w);
ih = ImageHelper();

patch_base_size = 16;
patches = ih.GeneratePatches(imgData(:,:,:), 10, patch_base_size);

count_patches = size(patches, 1);


ia = IA(0.7, 1.4);
ia.LearningRate = 0.02;
ia.UseGPU = false;

ch = ClassificationHelper();
data = ih.GeneratePatches(imgData(:,:,:), 10, patch_base_size);


pooling = @(x) Pooling(x, patch_base_size);
filter1 = @(x) ConcatClass(pooling(x), 1, 2);
filter2 = @(x) ConcatClass(pooling(x), 2, 2);

scoreFilter = @(x) zscore(x')';

input1 = ia.NewFilterNode((patch_base_size/2) ^2+2, filter1);
input2 = ia.NewFilterNode((patch_base_size/2) ^2+2, filter2);

n_centroids = 384;
% im1 = ia.NewIntentionalModule(input1.index, count_patches, 2, n_centroids); 
out1 = ia.NewOutputModule([input1.index, input2.index], count_patches*2, 4, n_centroids);
out2 = ia.NewOutputModule(input1.index, count_patches*2, 2, n_centroids);
% out3 = ia.NewOutputModule(input1.index, count_patches, 1, n_centroids);
% out4 = ia.NewOutputModule(input1.index, count_patches, 1, n_centroids);


while ia.IsBootstraping()
    input1.SetInput(patches);
    input2.SetInput(data);
    ia.Train();
end

ia.ClearInputOutput();

xx = 1;
current_image = reshape(imgData(xx,:,:), img_h, img_w);
current_image_rgb = uint8(current_image(:, :, [1 1 1]));

% [~, target_position] = ih.GeneratePatches(current_image, 1, patch_base_size);
target_position = floor([img_h/2, img_w/2]);

head_position = floor([(img_h - 1) * rand() + 1, (img_w -1) * rand() + 1]);
current_sigma = 20;

figure(9000);
colormap default;
img_plot = imagesc(current_image_rgb);
axis([1 img_w 1 img_h]);


npatches = 64;
time = 0;
figure(213);
hold on;
pp = plot(0.5*ones(1, 2000));
cc = plot(0.5*ones(1, 20));
hold off;

axis([1400 2000 0 1]);
start_dist = 0;
episode_time = 150;

while 1
    time = time+1;
    npatches = ceil(current_sigma);
    [current_patches, patches_position] = ih.SampleRegion(current_image, npatches, patch_base_size, head_position, current_sigma);
    
    input1.SetInput(current_patches);
    
    ia.Update();
    
    head_movement = ia.GetModuleOutput(out1);
    
    head_position = head_position + head_movement * ([-1 0; 0 -1; 1 0; 0 1] * 3);
    head_position = max([head_position; 1, 1 ], [], 1);
    head_position = min([head_position; img_h, img_w ], [], 1);
    
    head_focus = ia.GetModuleOutput(out2);
    current_sigma = current_sigma + head_focus * [1; -1];
    current_sigma = min(max(5, current_sigma), 10);
    
    
    RGB = insertShape(current_image, 'circle', [fliplr(head_position), current_sigma], 'LineWidth', 2, 'Color', 'red');
    RGB = insertShape(RGB, 'Rectangle', [fliplr(patches_position), patch_base_size*ones(npatches, 2)], 'LineWidth', 1, 'Color', 'white');
    RGB = insertShape(RGB, 'circle', [fliplr(target_position), 1], 'LineWidth', 5);
    
    img_plot.CData = uint8(RGB);
    
    
    
    
    dist = sqrt(sum( (head_position - [img_h/2, img_w/2]) .^2 ));
    
    
    if dist < 20
        xx = xx + (xx < 8);

        ia.NewEpisode(episode_time - time);
        time = 0;

        head_position = floor([(img_h - 1) * rand() + 1, (img_w -1) * rand() + 1]);
        start_dist = sqrt(sum((head_position - target_position).^2));
        
        cc.YData(1:end-1) = cc.YData(2:end);
        cc.YData(end) = 1;
        
        pp.YData(1:end-1) = pp.YData(2:end);
        pp.YData(end) = mean(cc.YData);
        continue;
    end
    
    if time > episode_time
        time = 0;
        ia.NewEpisode(-dist);

        head_position = floor([(img_h - 1) * rand() + 1, (img_w -1) * rand() + 1]);
        start_dist = sqrt(sum((head_position - target_position).^2));
        
        cc.YData(1:end-1) = cc.YData(2:end);
        cc.YData(end) = 0;
        
        pp.YData(1:end-1) = pp.YData(2:end);
        pp.YData(end) = mean(cc.YData);
        
        continue;
    end
    
    pause(0.001);
end
