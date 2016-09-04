classdef ImageHelper < handle
    %IMAGEHELPER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function ih = ImageHelper()
        
        end
        
        function [patches, positions] = GeneratePatches(~, images, patches_per_image, patch_size)
            images = double(images);
            
            if size(images,3) == 3
                ss = size(images);
                images = reshape(images, 1, ss(1), ss(2), ss(3));
            end
            
            if ismatrix(images)
                ss = size(images);
                images = reshape(images, 1, ss(1), ss(2));
            end
            
            ss = size(images);
            
            nimages = ss(1);
            img_h = ss(2);
            img_w = ss(3);
            
            pos = rand(nimages, patches_per_image, 2);
            
            pos(:,:,1) = floor((img_h - patch_size  ) * pos(:,:,1) + 1);
            pos(:,:,2) = floor((img_w - patch_size  ) * pos(:,:,2) + 1);
            
            patches = zeros(nimages * patches_per_image, patch_size ^2, 3 * (size(images,4) == 3) + (size(images,4) ~= 3));
            
            positions = zeros(nimages * patches_per_image, 2);

            for jj = 1:nimages
                img = reshape(images(jj,:,:), img_h, img_w, []);
                
                for ii = 1:patches_per_image
                    patch = img(pos(jj,ii, 1):pos(jj,ii, 1)+patch_size-1, pos(jj, ii, 2):pos(jj, ii, 2)+patch_size-1,:);
                    patches( (jj-1) * patches_per_image + ii, :,:) = reshape(patch, 1, patch_size ^2, []);
                    positions( (jj-1) * patches_per_image + ii, :) = ceil([pos(jj,ii, 1) + patch_size/2, pos(jj,ii, 2) + patch_size/2]);
                end
            end
        end
        
        function PlotPatches(~, patches, patch_w, patch_h, figure_no)
            
            plotted_patches = patches;

            n_patches = size(plotted_patches, 1);

            ncols = 32;
            nrows = floor(n_patches / ncols) + 1;

            figure(figure_no);
            
            for ii = 1:nrows
                for jj = 1:ncols
                    ind = (ii-1) * ncols + jj;

                    if ind > n_patches
                        break;
                    end

                    subplot(nrows, ncols, ind);
                    imagesc(reshape(plotted_patches(ind,:), patch_w, patch_h, []));
                    set(gca,'XTickLabel',{});
                    set(gca,'YTickLabel',{});
                end
            end 
        end
        
        function [patches, pos] = SampleRegion(~, image, npatches, patch_size, point, sigma)
            
            img_h = size(image, 1);
            img_w = size(image, 2);
            
            half_patch_size = ceil(patch_size/2);
            
            mu = repmat(point, npatches, 1);
            
            sigma = [sigma, sigma];
            sigma = repmat(sigma, npatches, 1);
            
            pos = ceil(normrnd(mu, sigma, npatches, 2) - half_patch_size);
            
            pos(pos < 1) = 1;
            pos(pos(:,1) + patch_size > img_h, 1) = img_h - patch_size;
            pos(pos(:,2) + patch_size > img_w, 2) = img_w - patch_size;
            
            patches = zeros(npatches, patch_size ^2);
            
            for ii = 1:npatches
                patch = image(pos(ii, 1):pos(ii, 1) + patch_size-1, pos( ii, 2):pos(ii, 2)+patch_size-1);
                patches(ii, :) = reshape(patch, 1, patch_size ^2);
            end
            
        end
        
        function img = AddCircle(~, image, pos_wh, radius, line_width, color)
            
            ww = 2;
            cc = 'black';
            
            if nargin() > 4
                ww = line_width;
                if nargin() > 5
                    cc = color;
                end
            end
            
            
            img = insertShape(image, 'circle', [pos_wh, radius], 'LineWidth', ww, 'Color', cc);
        end
        
        function img = AddRectangle(~, image, pos_wh, width, height, line_width, color)
            
            ww = 1;
            cc = 'black';
            
            if nargin() > 5
                ww = line_width;
                if nargin() > 6
                    cc = color;
                end
            end
            img = insertShape(image, 'Rectangle', [pos_wh, width, height], 'LineWidth', ww, 'Color', cc);
        end
        
        function img = AddCross(~, image, pos_wh, width, height, line_width, color)
             
            ww = 1;
            cc = 'black';
            
            if nargin() > 5
                ww = line_width;
                if nargin() > 6
                    cc = color;
                end
            end
            
            xy1 = pos_wh + [-width/2, -height/2];
            xy2 = pos_wh + [width/2, height/2];
            
            
            img = insertShape(image, 'Line', [xy1, xy2], 'LineWidth', ww, 'Color', cc);
            
            xy1 = pos_wh + [-width/2, height/2];
            xy2 = pos_wh + [width/2, -height/2];
            img = insertShape(img, 'Line', [xy1, xy2], 'LineWidth', ww, 'Color', cc);
        
        end
    end
    
end

