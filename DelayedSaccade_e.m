
img_h = 120;
img_w = 160;
%CREATE RANDOM IMAGES
nimages = 800;
imgData = zeros(nimages, img_h, img_w);
ih = ImageHelper();

for ii = 1:nimages
    image = 255 * ones(img_h, img_w);
    
    n_shapes = 20;
    
    for jj = 1:n_shapes
        coords = ceil(rand(1,2) .* [img_h, img_w]);
        line_width = ceil( 5*rand() );
        edge_size = ceil( 40*rand() );
        image = ih.AddCross(image, coords, edge_size, edge_size, line_width, 'black');
    end
    
        coords = ceil(rand(n_shapes,2) .* repmat([img_h, img_w], n_shapes, 1));
        line_width = ceil( 5*rand() );
        radius = ceil( 40*rand(n_shapes,1) );
        image = ih.AddCircle(image, coords, radius, line_width, 'black');
    
    imgData(1,:,:) = rgb2gray(image);
end



% load('/home/michele/Development/IDRA/IDRA/data/imgData.mat')

imgData = double(imgData);

ih = ImageHelper();


patch_base_size = 32;
npatches = 120;

patches = ih.GeneratePatches(imgData, npatches, patch_base_size);


count_patches = size(patches, 1);


ia = IA(0.7, 1.4);
ia.ExplorationPercentage = 0;
ia.LearningRate = 0.02;
ia.UseGPU = false;

normFilter = @(x) Normalize(x, 0, 255);
pooling = @(x) Pooling(x, patch_base_size);

n_inputs = 4;
filter1 = @(x) ConcatClass(pooling(x), 1, n_inputs);
filter2 = @(x) ConcatClass(pooling(x), 2, n_inputs);
filter3 = @(x) ConcatClass(pooling(x), 3, n_inputs);
filter4 = @(x) ConcatClass(pooling(x), 4, n_inputs);




input1 = ia.NewFilterNode((patch_base_size/2) ^2 + n_inputs, filter1);
input2 = ia.NewFilterNode((patch_base_size/2) ^2 + n_inputs, filter2);
input3 = ia.NewFilterNode((patch_base_size/2) ^2 + n_inputs, filter3);
input4 = ia.NewFilterNode((patch_base_size/2) ^2 + n_inputs, filter4);

n_centroids = 8;
out = ia.NewOutputModule([input1.index, input2.index, input3.index, input4.index], count_patches*4, 4, n_centroids);



while ia.IsBootstraping()
    input1.SetInput(patches);
    input2.SetInput(patches);
    input3.SetInput(patches);
    input4.SetInput(patches);
    ia.Train();
end
ia.ClearInputOutput();


current_image = reshape(imgData(1,:,:), img_h, img_w);




figure(9000);
colormap default;
img_plot = imagesc(current_image);
axis([1 img_w 1 img_h]);


figure(213);
hold on;
pp = plot(0.5*ones(1, 2000));
cc = plot(0.5*ones(1, 2000));
hold off;


heads_position = [30, 30; 30, img_h-30; img_w-30, 30; img_w-30, img_h-30;];
gaze_position = ceil([img_w/2, img_h/2]);
current_sigma = 15;
npatches = 20;
episode_time = 0;
episode_max_time = 10;
phase = 1;

rew_position = ceil(rand() * 4);

while 1
    
    
    current_image = ones(img_h, img_w);
    
    if phase == 1 %showing reward position
        no_rew_pos = heads_position(1:end ~= rew_position,:);
        
        for kk = 1:size(no_rew_pos, 1)
            current_image = ih.AddCross(current_image, no_rew_pos(kk,:), 20, 20, 3, 'black');
        end
        
        current_image = ih.AddCircle(current_image, heads_position(rew_position,:), 10, 3, 'black');
        
    end
    
    if phase == 2
        n_crosses = 20;
        for kk = 1:size(no_rew_pos, 1)
            current_image = ih.AddCross(current_image, ceil( rand(1,2) .* [img_h, img_w]), 20, 20, 3, 'black');
        end
    end
    
    % SAMPLE CORNER REGIONS
    [current_patches, ~] = ih.SampleRegion(current_image, npatches, patch_base_size, heads_position(1,:), current_sigma);
    input1.SetInput(current_patches);
    
    
    [current_patches, ~] = ih.SampleRegion(current_image, npatches, patch_base_size, heads_position(2,:), current_sigma);
    input2.SetInput(current_patches);
    
    [current_patches, ~] = ih.SampleRegion(current_image, npatches, patch_base_size, heads_position(3,:), current_sigma);
    input3.SetInput(current_patches);
    
    [current_patches, ~] = ih.SampleRegion(current_image, npatches, patch_base_size, heads_position(4,:), current_sigma);
    input4.SetInput(current_patches);
    %
    
    
    
    ia.Update();
    
    % UPDATE GAZE POSITION
    module_out = ia.GetModuleOutput(out);
    
    gaze_position = gaze_position + module_out * ([-1 -1; 1 -1; -1 1; 1 1] * 5);
    gaze_position = max([gaze_position; 1, 1 ], [], 1);
    gaze_position = min([gaze_position; img_h, img_w ], [], 1);
    
    
    current_image = insertShape(current_image, 'circle', [gaze_position, current_sigma], 'LineWidth', 2, 'Color', 'red');
    current_image = rgb2gray(current_image);
    
    img_plot.CData = uint8(current_image);
    
    
    
    
    dist = sqrt(sum( (gaze_position - heads_position(rew_position,:)) .^2 ));
    
    
    if dist < 35
        ia.NewEpisode(20);
        ia.ClearInputOutput();
        episode_time = 0;

        gaze_position = ceil([img_w/2, img_h/2]);
        rew_position = ceil(rand() * 4);
        
        cc.YData(1:end-1) = cc.YData(2:end);
        cc.YData(end) = 1;
        
        pp.YData(1:end-1) = pp.YData(2:end);
        pp.YData(end) = mean(cc.YData(end-100:end));
        
    end
    
    if episode_time > episode_max_time
        ia.NewEpisode(-20);
        ia.ClearInputOutput();
        episode_time = 0;

        gaze_position = ceil([img_w/2, img_h/2]);
        rew_position = ceil(rand() * 4);
        
        
        cc.YData(1:end-1) = cc.YData(2:end);
        cc.YData(end) = 0;
        
        pp.YData(1:end-1) = pp.YData(2:end);
        pp.YData(end) = mean(cc.YData(end-100:end));
        
        
    end
    
    
    if episode_time <= 5
        phase = 1;
    end
    
    if episode_time > 5
        phase = 2;
    end
    
    episode_time = episode_time + 1;
    
    pause(0.001);
end