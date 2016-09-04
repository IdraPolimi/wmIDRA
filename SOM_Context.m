classdef SOM_Context < handle
    properties 
        map
        count_receptors
        
        neighborhood_size
        
        current_activation
        
        map_indeces
        
        LastHit
        UseGPU
        
        graph
        current_input_graph
        
    end
    
    methods
        function ctx = SOM_Context(size_h, size_w, max_categories, neighborhood_size)
            ctx.neighborhood_size = neighborhood_size;
            
            ctx.count_receptors = 1;
            
            ctx.map = zeros(size_h, size_w, max_categories);
            
            ctx.map_indeces = zeros(size_h, size_w, 2);
            
            indeces_h = (1:size_w);
            indeces_h = repmat(indeces_h, size_h, 1);
            
            indeces_w = (1:size_h)';
            indeces_w = repmat(indeces_w, 1, size_w);
            
            ctx.map_indeces(:, :, 1) = indeces_h;
            ctx.map_indeces(:, :, 2) = indeces_w;
            
            
            ctx.current_activation = zeros(1, max_categories);
            ctx.LastHit = zeros(1, 2);
            
            ctx.UseGPU = 0;
            
            
            figure(2764);
            hold on;
%             ctx.graph = image(zeros(size_h, size_w), 'CDataMapping','scaled');
            axis([0 size_w 0 size_h]);
            ctx.current_input_graph = scatter(0,0,'*', 'R');
            hold off;
        end
        
        function DoubleSize(ctx)
            
            [width, height] = ctx.SOM_Width_Height();
            
            curr_size = length(ctx.current_activation);
            new_size = 2 * curr_size;
            inc_size = new_size - curr_size;
            
            ctx.current_activation = cat(2, ctx.current_activation, zeros(1, inc_size));
            
            
            curr_map_size = size(ctx.map, 3);
            new_map_size = 2 * curr_size;
            inc_map_size = new_map_size - curr_map_size;
            
            ctx.map = cat(3, ctx.map, zeros(height, width, inc_map_size));
        end
        
        function indeces = NextAvailableIndeces(ctx, amount)
           
           while ctx.CountReceptors() + amount >= ctx.MaxReceptors();
               ctx.DoubleSize();
           end
            
           indeces = zeros(amount, 1);
           for ii = 1:amount
               indeces(ii) = ctx.count_receptors;
               ctx.count_receptors = ctx.count_receptors +1;
           end
           
           
        end
        
        function receptor = AddReceptor(ctx)
            index = ctx.NextAvailableIndeces(1);
            receptor = SOM_Receptor(ctx, index);
        end
    
        function Update(ctx)
            
            [width, height] = ctx.SOM_Width_Height();
            useGPU = ctx.UseGPU();
            
            current_input = ctx.current_activation;
            dists = zeros(height, width);
            current_map = ctx.map;
            
            if useGPU
                current_input = gpuArray(current_input);
                dists = gpuArray(dists);
                current_map = gpuArray(current_map);
            end
            
            % we find BMU
            current_input = repmat(current_input, [height, 1, width]);
            current_input = permute(current_input, [1 3 2]);
            
            dists = (current_map - current_input).^2;
            
            dists = sqrt(sum(dists, 3));
            
            
            [bmu_col, bmu_row] = find(dists == min(dists(:)));
            
            
            bmu_col = bmu_col(randsample(length(bmu_col), 1));
            bmu_row = bmu_row(randsample(length(bmu_row), 1));
            
            
            % we generate the neighborhood mask
            indeces = ctx.map_indeces;
            nh_dist = cat(3, repmat(bmu_row, height, width), repmat(bmu_col, height, width));
            
            nh_dist = (indeces - nh_dist).^2;
            nh_dist = sqrt(sum(nh_dist,3));
            
            nh = ctx.neighborhood_size;
            
            
            mask = nh_dist <= ctx.neighborhood_size;
            
            nh_dist = mask.*nh_dist;
            
            mean_nh_dist = mean(mean(nh_dist));
            
            update_mask = nh_dist;
            update_mask(update_mask == 0) = Inf;
            update_mask(bmu_col, bmu_row) = 0;
            
            % update map
            
            update_rate = 0.33 * gaussmf(update_mask,[mean_nh_dist/sqrt(2*log(2)) 0]);
            
            update_rate = repmat(update_rate, 1, 1, size(current_map, 3));
            
            if useGPU
                current_input = gather(current_input);
                bmu_col = gather(bmu_col);
                bmu_row = gather(bmu_row);
                update_rate = gather(update_rate);
            end
            
            ctx.map = update_rate .* current_input + (1 - update_rate) .* ctx.map; 
            
            
            ctx.graph.CData = ctx.GetPMatrix();
            
            ctx.current_input_graph.XData = bmu_row;
            ctx.current_input_graph.YData = bmu_col;
            
            ctx.LastHit = [bmu_col, bmu_row];
        end
        
        function res = CountReceptors(ctx)
            res = ctx.count_receptors;
        end
        
        function res = MaxReceptors(ctx)
            res = size(ctx.map, 3);
        end
        
        function pmat = GetPMatrix(ctx)
            som = ctx.map;
            som_dim = size(som);
            data = zeros(som_dim(1), som_dim(2));
            tmp = zeros(som_dim(1), som_dim(2),2);



            %----- 1
            kern = [0, -1, 1];
            for ii = 1:som_dim(3)

                cc = conv2(som(:,:,ii),kern);
                tmp(:,:,ii) = cc(:,3:end);


            end
            tmp = tmp .^2;
            data = data + sqrt(sum(tmp,3));

            %----- 2
            kern = [1, -1, 0];
            for ii = 1:som_dim(3)
                cc = conv2(som(:,:,ii),kern);
                tmp(:,:,ii) = cc(:,1:end-2);
            end
            tmp = tmp .^2;
            data = data + sqrt(sum(tmp,3));

            %----- 3
            kern = [0; -1; 1];
            for ii = 1:som_dim(3)
                cc = conv2(som(:,:,ii),kern);
                tmp(:,:,ii) = cc(3:end,:);
            end
            tmp = tmp .^2;
            data = data + sqrt(sum(tmp,3));

            %----- 4
            kern = [1; -1; 0];
            for ii = 1:som_dim(3)
                cc = conv2(som(:,:,ii),kern);
                tmp(:,:,ii) = cc(1:end-2,:);
            end
            tmp = tmp .^2;
            data = data + sqrt(sum(tmp,3));

            %---- normalize
            data = data ./4;

            data(1,:) = 0;
            data(end,:) = 0;
            data(:,1) = 0;
            data(:,end) = 0;

            pmat = data;
        end
        
    end
    
    methods 
        
        function [w, h] = SOM_Width_Height(ctx)
            [h, w, ~] = size(ctx.map);
        end
        
        function res = get.UseGPU(ctx)
        	res = ctx.UseGPU;
        end
        
        function set.UseGPU(ctx, val)
        	try
                gpuDevice;
            	ctx.UseGPU = val ~= 0;
            catch
                ctx.UseGPU = false;
        	end
        end
    end
    
    methods (Access = public)
        
        function activations = GetActivations(ctx, indeces)
            
            fields = ctx.map(:,:,indeces);
            s1 = size(indeces,1);
            s2 = size(indeces, 2);
            
            mm = max(max(fields));
            activations = reshape(mm, s1, s2);
            
        end
        
        function SetActivations(ctx, indeces, values)
            ctx.current_activation(indeces) = values;
        end
    end
end
