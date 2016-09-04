classdef IA < handle
    
    properties ( Access = private )
       TYPE_INPUT = 1
       TYPE_CONTEXT = 2
       TYPE_IM = 3
       TYPE_BOOTSTRAPING = 4
       TYPE_OUT = 5
       TYPE_EMPTY = 0
    end
    
    
    properties 
        modules
        modules_type
        next_index
        connections
        
        threshold
        graphs
        network_graph
        network_graph_count
        network_graph_plot
        network_graph_fig_no
        
        
        gain
        centroids_multiplier
        
        sampling_percentage
        
        ExplorationPercentage
        LearningRate
        wm
    end
    
    properties
        UseGPU
        im_ca
    end
    
    methods
        
        function ia = IA(treshold, gain)
            size = 1;
            
            ia.UseGPU = 0;
            
            ia.modules = cell(size,1);
            ia.next_index = 1;
            ia.modules_type = zeros(size,1);
            ia.connections = zeros(size, 1);
            ia.threshold = treshold;
            ia.gain = gain;
            
            ia.centroids_multiplier = 1;
            
            ia.sampling_percentage = 1;
            
            ia.wm = WM();
            ia.wm.init();
            ia.ExplorationPercentage = 0.1;
            ia.LearningRate = 0.002;
            ia.graphs = cell(size,1);
            
            
            count = ia.CountModules();
            ia.network_graph = digraph(ia.connections(1:count, 1:count));
            ia.network_graph_count = 0;
            ia.network_graph_fig_no = 42;
            
            figure(ia.network_graph_fig_no);
            ia.network_graph_plot = plot(ia.network_graph);
        end
        
        function set.UseGPU(ia, val)
           ia.UseGPU = val;
        end
        
        function set.ExplorationPercentage(ia, percentage)
            ia.ExplorationPercentage = percentage;
        end
        
        function perc = get.ExplorationPercentage(ia)
            perc = ia.ExplorationPercentage;
        end
        
        function set.LearningRate(ia, rate)
            ia.LearningRate = rate;
        end
        
        function rate = get.LearningRate(ia)
            rate = ia.LearningRate;
        end
        
        function c = CountModules(ia)
            c = ia.next_index - 1;
        end
        
        function s = MaxSize(ia)
           s = length(ia.modules); 
        end
        
        function input_module = NewFilterNode(ia, input_size, filterFunction)
            index = ia.NextIndex(ia.TYPE_INPUT);
            module = [];
            module.input = zeros(1, input_size);
            module.output = zeros(1, input_size);
            module.activation = 1;
            module.reward = 0;
            module.output_changed = false;
            
            ia.modules{index} = module;
            input_module = FilterNode(ia, index, input_size, filterFunction);
        end
        
        function index = NewIntentionalModule(ia, sources, training_set_size, n_retained_chunks, n_centroids)
            index = ia.NextIndex(ia.TYPE_BOOTSTRAPING);
            ia.connections(index, sources) = 1;
            input_size = ia.GetModuleOutputSize(sources);
            
            
            module = [];
            module.index = index;
            module.sources = sources;
            module.input = zeros( 1, input_size );
            module.output = zeros( 1, input_size );
            module.n_centroids = n_centroids;
            module.similarities = zeros( 1, n_centroids );
            module.threshold = ia.threshold;
            module.bootstraping = true;
            module.training_set = zeros(training_set_size, input_size);
            module.sampling_weights = ones(training_set_size,1);
            module.training_count = 0;
            module.ica = [];
            module.pca = [];
            module.centroids = [];
            module.centroids_mean_distance = [];
            module.activation = 0;
            module.output_changed = false;
            module.output_module = false;
            
            module.n_retained_chunks = n_retained_chunks;
            
            module.update = false;
            module.reward = 0;
            
            module.time = 0;
            
            ia.modules{index} = module;
        end
        
        function index = NewTrainedIntentionalModule(ia, sources, training_set, n_retained_chunks, n_centroids)
            tset_size = size(training_set, 1);
            index = ia.NewIntentionalModule(sources, tset_size, n_retained_chunks, n_centroids);
            
            ia.UpdateTrainingSet(index, training_set);
            ia.TrainIntentionalModule(index);
        end
        
        function index = NewOutputModule(ia, sources, training_set_size, n_actions, n_centroids)
            index = ia.NewIntentionalModule(sources, training_set_size, 1, n_centroids);
            
            ia.modules{index}.n_actions = n_actions;
            ia.modules{index}.output_module = true;
            ia.modules{index}.output = zeros( 1, n_actions );
        end
        
        function bs = IsBootstraping(ia, indeces)
            if nargin <= 1
                bs = any(ia.IsBootstraping(1:ia.CountModules()));
            else
                bs = ia.modules_type(indeces);

                bs = bs == ia.TYPE_BOOTSTRAPING;
            end
        end
        
        function Update(ia)
            ia.UpdateInputModules();
            ia.UpdateIntentionalModules();
            ia.UpdateWorkingMemory();
            ia.UpdateOutputModules();
            ia.UpdateGraph();
        end
        
    end
    
    methods
        
        function size = GetModuleOutputSize(ia, indeces)
            size = 0;
            for ii = 1:length(indeces)
                index = indeces(ii);
                size = size + length(ia.modules{index}.output);
            end
        end
        
        function size = GetModuleInputSize(ia, indeces)
            size = 0;
            for ii = 1:length(indeces)
                index = indeces(ii);
                size = size + length(ia.modules{index}.input);
            end
        end
        
        function out = GetModuleOutput(ia, indeces)
            out = [];
            for ii = 1:length(indeces)
                index = indeces(ii);
                out = cat(2, out, ia.modules{index}.output);
            end
        end
        
        function rewards = GetModuleReward(ia, indeces)
            rewards = [];
            for ii = 1:length(indeces)
                index = indeces(ii);
                rewards = cat(2, rewards, ia.modules{index}.reward);
            end
        end
        
        function SetModuleInput(ia, index, values)
           ia.modules{index}.input = values;
        end
        
        function parents = GetInputModules(ia, indeces)
            parents = ia.GetParentModules(indeces);
        end
        
        function childrens = GetChildrenModules(ia, indeces)
            childrens = find( any(ia.connections(:,indeces), 2) == true );
        end
        
        function parents = GetParentModules(ia, indeces)
            parents = find(ia.connections( indeces,:) == 1);
        end
        
        function siblings = GetSiblingModules(ia, indeces)
           parents = ia.GetParentModules(indeces);
           siblings = ia.GetChildrenModules(parents);
           siblings = siblings(siblings ~= indeces);
        end
        
        function layers_count = CountLayers(ia)
            inputs = find(all(ia.connections' == 0, 1));
            
            layers_count = 0;
            
            childrens = ia.GetChildrenModules(inputs);
            
            while ~isempty(childrens)
                layers_count = layers_count + 1;
                childrens = ia.GetChildrenModules(childrens);
            end
        end
        
        function booting = AreChildrenBootstraping(ia,index)
            childrens = ia.GetChildrenModules(index);
            
            booting = any( ia.IsBootstraping(childrens) );
        end
        
        function UpdateInputModules(ia)
            indeces = find(ia.modules_type == ia.TYPE_INPUT);
            
            for ii = 1:length(indeces)
                index = indeces(ii);
                module = ia.modules{index};
                
                module.output = module.input;
                
                ia.modules{index} = module;
            end
        end
        
        function UpdateWorkingMemory(ia)
            indeces = find(ia.modules_type == ia.TYPE_IM);
            
            
            for ii = 1:length(indeces)
                index = indeces(ii);
                module = ia.modules{index};
                
                
                bootstraping = ia.IsBootstraping();
                if bootstraping
                   ia.wm.SetExplorationPercentage(module.wm_index, 1);
                   ia.wm.SetLearningRate(module.wm_index, 0);
                else
                	ia.wm.SetLearningRate(module.wm_index, ia.LearningRate);
                    ia.wm.SetExplorationPercentage(module.wm_index, ia.ExplorationPercentage);
                end
                
                update = module.state_changed || module.time > 20 || bootstraping;
                module.update = false;
                
                if  update
                    module.time = 0;
                    
                    ia.wm.SetReward(module.wm_index, module.reward);
                    module.reward = 0;
                    
                    ia.wm.SetState(module.wm_index, zeros(1));
                   
                    weights = eye(module.ica.n_ics) .* repmat(module.ica_weights, module.ica.n_ics, 1);

                    chunks = zeros(module.ica.n_ics, module.n_centroids*module.ica.n_ics);

                    max_s = module.max_simil_index;
                    cols = ((max_s - 1) * module.ica.n_ics + 1):(module.ica.n_ics*max_s);
                    chunks(:, cols) = weights;

                    ia.wm.SetCandidateChunks(module.wm_index, chunks);
                    ia.wm.EpisodeTick(module.wm_index);

                    retained_chunks = ia.wm.GetRetainedChunks(module.wm_index);
                
                    module.output_changed = false;
                    if isempty(retained_chunks)
                        module.out_mask = zeros(1, module.ica.n_ics);
                        module.output =  (module.raw_weights * module.ica.ics) * module.pca.coeff' + ...
                                          repmat(module.pca.mu, size(module.output, 1), 1);
                    else
                        
                        
                        retained_chunks = sum(retained_chunks, 1);
                        
                        retained_chunks = reshape(retained_chunks, module.ica.n_ics, module.n_centroids);
                        
                        
                        new_mask = sum(retained_chunks, 2)';
                        
                        if any((new_mask ~= 0) ~= (module.out_mask ~= 0))
                            module.out_mask = new_mask;
                            module.output_changed = true;
                        end
                        mraw = module.raw_weights;
                        mraw(module.out_mask == 0) = 0;
                        module.output =  (mraw * module.ica.ics) * module.pca.coeff' + ...
                                          repmat(module.pca.mu, size(module.output, 1), 1);
                    end
%                     figure(45875);
%                     plot(module.input)
%                     hold on
%                     plot(module.output)
%                     hold off;
                end
                ia.modules{index} = module;
            end
        end
        
        function UpdateOutputModules(ia)
            indeces = find(ia.modules_type == ia.TYPE_OUT);
            
            for ii = 1:length(indeces)
                index = indeces(ii);
                module = ia.modules{index};
                
                
                bootstraping = ia.IsBootstraping();
                if bootstraping
                   ia.wm.SetExplorationPercentage(module.wm_index, 1);
                   ia.wm.SetLearningRate(module.wm_index, 0);
                else
                	ia.wm.SetLearningRate(module.wm_index, ia.LearningRate);
                    ia.wm.SetExplorationPercentage(module.wm_index, ia.ExplorationPercentage);
                end
                
                update = module.state_changed || module.time > 20 || bootstraping;
                module.update = false;
                
                if  update
                    module.time = 0;
                    
                    
                    ia.wm.SetReward(module.wm_index, module.reward);
                    module.reward = 0;
                    
                    ia.wm.SetState(module.wm_index, zeros(1));
                    
                    
                    chunks = zeros(module.n_actions, module.n_centroids * module.n_actions);

                    max_s = module.max_simil_index;
                    cols = ((max_s - 1) * module.n_actions + 1):(module.n_actions*max_s);
                    chunks(:, cols) = eye(module.n_actions);
                    
                    ia.wm.SetCandidateChunks(module.wm_index, chunks);
                    ia.wm.EpisodeTick(module.wm_index);

                    retained_chunks = [];
                        
                    retained_chunks = ia.wm.GetRetainedChunks(module.wm_index);
                    
                    module.output_changed = false;
                    if isempty(retained_chunks)
                    	module.out_mask = zeros(1, module.n_actions);
                    else
                        max_s = find(retained_chunks);
                        mod_s = mod(max_s, module.n_actions);
                        max_s = max_s - mod_s;
                        max_s = max_s / module.n_actions + double(mod_s > 0);
                        cols = ((max_s - 1) * module.n_actions + 1):(module.n_actions*max_s);
                        new_mask = sum(retained_chunks(:,cols), 1);
                        if any((new_mask ~= 0) ~= (module.out_mask ~= 0))
                            module.out_mask = new_mask;
                            module.output_changed = true;
                        end
                    end
                    %--
                    module.output =  module.out_mask;
                    
                end
                ia.modules{index} = module;
            end
        end
        
        function UpdateIntentionalModules(ia)
            indeces = find(ia.modules_type == ia.TYPE_BOOTSTRAPING);
            
            for ii = 1:length(indeces)
                index = indeces(ii);
                sources = ia.GetParentModules(index);
                
                sa = ia.GetModuleActivation(sources);


                if any(ia.IsBootstraping(sources))
                    continue;
                end

                if max(sa) < ia.threshold
                    continue;
                end
                
                ia.TrainIntentionalModule(index);
            end
            
            indeces = find(ia.modules_type == ia.TYPE_IM);
            indeces = cat(1, indeces, find(ia.modules_type == ia.TYPE_OUT));
            
            for ii = 1:length(indeces)
                index = indeces(ii);
                module = ia.modules{index};
                
                input = ia.GetModuleOutput(module.sources);
                module.input = input;
                
                [module.ica_weights, module.in_pca, module.raw_weights] = ia.ProcessForward(module, input);
                
                similarity = ia.GetCentroidsDistance(module);
                simsum = max(sum(similarity, 2), 10^-16);

                similarity = similarity ./ simsum;

                
                module.state_changed = any( (module.similarities == max(module.similarities)) ~=  ...
                                            (similarity == max(similarity)) ...
                                            );
                
                module.similarities = similarity;
                
                module.max_simil_index = find(module.similarities == max(module.similarities));
                module.max_simil_index = module.max_simil_index( floor( rand() * length(module.max_simil_index) ) + 1 );
                
                module.activation = max(similarity);
                
                
%                 ia.graphs{index}.output.XData = module.in_pca(1);
%                 ia.graphs{index}.output.YData = module.in_pca(2);
%                 ia.graphs{index}.output.ZData = module.in_pca(3);

                module.time = module.time + 1;
                ia.modules{index} = module;
                
            end
            
        end
        
        function full = UpdateTrainingSet(ia, index, input)
            
           if all(input == 0)
               full = false;
           end
            
           module = ia.modules{index};
           curr_count = module.training_count;
           max_count = size(module.training_set, 1);
           input_length = size(input, 1);
           
           
           if curr_count < max_count
               new_count = min(curr_count + input_length, max_count);
               ia.modules{index}.training_set(curr_count + 1:new_count, :) = input(1:new_count-curr_count, :);
               ia.modules{index}.training_count = new_count;
               
               if new_count == max_count
                   full = true;
               else
                   full = false;
               end
               
           else
              full = true; 
           end
        end
        
        function UpdateGraph(ia)
            
            count = ia.CountModules();
            
            if count ~= ia.network_graph_count
                ia.network_graph_count = count;
                ia.network_graph = digraph(ia.connections(1:count, 1:count));

                figure(ia.network_graph_fig_no);
                ia.network_graph_plot = plot(ia.network_graph);
                highlight(ia.network_graph_plot, 1:count);
            end
            
            highlight(ia.network_graph_plot,ia.network_graph,'EdgeColor',[0 0.4470 0.7410],'LineWidth',0.33);
            
            max_active_adj = ia.connections(1:count, 1:count);
            
            for ii = 1:count
                
                pp = max(0, min(ia.modules{ii}.activation,1));
                cc = 0;
                if ia.modules{ii}.output_changed
                    cc = 0.9;
                end
                highlight(ia.network_graph_plot, ii, 'NodeColor', [1 - pp, pp, cc]);
                
                if ~ia.modules{ii}.output_changed
                    max_active_adj(:,ii) = 0;
                end
                
            end
            
            h = digraph(max_active_adj);
            highlight(ia.network_graph_plot, h, 'EdgeColor','black','LineWidth',2)
        end
        
        function NewEpisode(ia, reward)
            indeces = find(ia.modules_type == ia.TYPE_IM);
            indeces = cat(1, indeces, find(ia.modules_type == ia.TYPE_OUT));
            
            for ii = 1:length(indeces)
                index = indeces(ii);
                module = ia.modules{index};
                
                module.similarities = zeros(size(module.similarities));
                module.input = zeros(size(module.input));
                module.output = zeros(size(module.output));
                
                reward = reward + module.reward;
                module.reward = 0;
                
                ia.wm.SetReward(module.wm_index, reward );
                ia.wm.EpisodeTick(module.wm_index);
                
                ia.wm.SetState(module.wm_index, zeros(1));
                ia.wm.NewEpisode(module.wm_index);
                ia.wm.SetReward(module.wm_index, 0 );
                
                module.update = true;
                ia.modules{index} = module;
            end
        end
        
        function IncrementReward(ia, reward)
            indeces = find(ia.modules_type == ia.TYPE_IM);
            indeces = cat(1, indeces, find(ia.modules_type == ia.TYPE_OUT));
            
            for ii = 1:length(indeces)
                index = indeces(ii);
                ia.modules{index}.reward = ia.modules{index}.reward + reward;
            end
            
        end
        
        function a = GetModuleActivation(ia, indeces)
            l = length(indeces);
            a = zeros(1, l);
            
            for ii = 1:l
                a(ii) = ia.modules{indeces(ii)}.activation;
            end
        end
        
        function TrainIntentionalModule(ia, index)
            module = ia.modules{index};
            
            sources = module.sources;
            
            
            input = ia.GetModuleOutput(sources);
            
%             
            if ia.UpdateTrainingSet(index, input)
                
                training_set = ia.modules{index}.training_set;
                training_set_size = size(training_set, 1);
                
                sampled_set_size = ceil(training_set_size * ia.sampling_percentage);
                
                w = ia.modules{index}.sampling_weights;
                
                samples = randsample(training_set_size, sampled_set_size, true,w);
                
                sampled_training_set = training_set(samples, :);
                sampled_training_set = sampled_training_set(randperm(size(sampled_training_set,1)),:);
                
                %---------------- MODULE TRAINING ------------------ begin
                sigin = sampled_training_set;
                
                %PCA
                [pc.coeff, pc.score, pc.latent, pc.tsquared, pc.explained, pc.mu] = pca(sigin);
                
                perc = cumsum(pc.latent) / sum(pc.latent);
                perc(perc > 0.999) = 0;
                perc(perc > 0 ) = 1;
                perc( 1:3 ) = 1;
                k = sum(perc);

                [pc.coeff, pc.score, pc.latent, pc.tsquared, pc.explained, pc.mu] = pca(sigin, 'NumComponents', k);
                
                pc.inv_coeff = pinv(pc.coeff');
                %ICA
                [ica.ics, ica.A, ica.W] = fastica(pc.score, 'numOfIc', 64);
                
                inv_ics = pinv(ica.ics);
                
                mod_inv_ics = repmat(sqrt( sum(inv_ics.^2, 1) ), size(inv_ics, 1), 1);
                
                ica.inv_ics = inv_ics;
                ica.mod_inv_ics = mod_inv_ics;
                ica.n_ics = size(ica.ics, 1);
                
                
                %--
                module.ica = ica;
                module.pca = pc;
                %--
                
                
                [~, in_pca, ~] = ia.ProcessForward(module, training_set);
                
                
                [centroids, ~, meand] = categorizeInput(in_pca, module.n_centroids);
                
                module.centroids = centroids;
                module.centroids_mean_distance = meand';
                module.bootstraping = false;
                
                if module.output_module
                    module.wm_index = ia.wm.NewWMModule(module.n_retained_chunks, 1, module.n_actions*module.n_centroids);
                    module.out_mask = zeros(1, module.n_actions);
                else
                    module.wm_index = ia.wm.NewWMModule(module.n_retained_chunks, 1, module.ica.n_ics*module.n_centroids);
                    module.out_mask = zeros(1, ica.n_ics);
                end
                ia.wm.SetLearningRate(module.wm_index, ia.LearningRate);
                ia.wm.SetExplorationPercentage(module.wm_index, ia.ExplorationPercentage);
                ia.wm.SetGamma(module.wm_index, 0.5);
                
                
                if module.output_module
                    ia.modules_type(index) = ia.TYPE_OUT;
                else
                    ia.modules_type(index) = ia.TYPE_IM;
                end
                
                ia.modules{index} = module;
                %---------------- MODULE TRAINING ------------------ end
                
            
            
                module = ia.modules{index};
                
                [~, current_output, ~] = ia.ProcessForward(module, input);
                figure(index * 13);
                
                hold on;
                
                plotted = in_pca;
                
                for aa = size(plotted,2)+1:3
                    plotted(:,aa) =  zeros(size(plotted,1),1);
                end
                for aa = size(centroids,2)+1:3
                    centroids(:,aa) = zeros(size(centroids,1),1);
                end
                for aa = size(current_output,2)+1:3
                    current_output(:,aa) = zeros(size(current_output,1),1);
                end
                
                ia.graphs{index}.training = scatter3(plotted(:,1 ),plotted(:, 2),plotted(:, 3), 'G', '.');
                ia.graphs{index}.centroids = scatter3(centroids(:,1 ),centroids(:, 2),centroids(:, 3), 'B');
                ia.graphs{index}.output = scatter3(current_output(:,1 ),current_output(:, 2),current_output(:, 3), 'R');
                hold off;
                axis auto;
                grid on;
                
            end
        end
        
        function [ica_weights, in_pca, raw_weights] = ProcessForward(~, module, inputs)
            
            inv_ics = module.ica.inv_ics;
            pcs = module.pca;
            mu = pcs.mu;
            inv_coeff = pcs.inv_coeff;
            
            in_pca = (inputs - repmat(mu, size(inputs,1), 1)) * inv_coeff;
            
            mod_in_pca = repmat(sqrt( sum(in_pca.^2, 2) ), 1, size(in_pca, 2));
            
            ica_weights = (in_pca ./ mod_in_pca) * (inv_ics ./ module.ica.mod_inv_ics);
            raw_weights = in_pca * inv_ics;
        end
       
        function sim = GetCentroidsDistance(ia, module)
            centroids = module.centroids;
            
            [~, points,~] = ia.ProcessForward(module, module.input);
            
            count_centroids = size(centroids, 1);
            count_points = size(points, 1);
            
            
            meand = module.centroids_mean_distance;
            
            dists = pdist(cat(1, points, centroids), 'euclidean');
            
            dists = dists(1:count_centroids * count_points);

            dists = reshape(dists, count_centroids, count_points)';

            c = max([meand; 10^-2*ones(size(meand))], [], 1);
            
            c = repmat(c, count_points, 1);
            
            b = 1./(2*ia.gain*c.^2);
            
            sim = exp(-b/ia.gain .* (dists.^2));

            %NaN means division by zero ie. point and centroid are the same
            sim(isnan(sim)) = 1;

        end
        
        function index = NextIndex(ia, nodeType)
            while ia.CountModules() >= ia.MaxSize()
                ia.DoubleSize();
            end
            
            index = ia.CountModules() + 1;
            ia.next_index = ia.next_index + 1;
            
            ia.modules_type(index) = nodeType;
        end
        
        function index = NewContextModule(ia, source_module_index)
            
            index = ia.NextIndex(ia.TYPE_CONTEXT);
            
            input_size = ia.GetModuleInputSize(source_module_index);
            output_size = ia.GetModuleOutputSize(source_module_index);
            module = [];
            
            module.source = source_module_index;
            module.input = zeros(input_size,1);
            module.output = zeros(output_size, 1);
            module.receptors = ia.context.NextAvailableIndeces( output_size )';
            module.activation = 0;
            
            module.cache_size = 1000;
            module.cache_count = 0;
            module.cache = zeros(module.cache_size, output_size);
            
            ia.modules{index} = module;
            
        end
        
        function DoubleSize(ia)
            
            curr_size = length(ia.modules);
            new_size = 2 * curr_size;
            increase = new_size - curr_size;
            
            ia.modules = cat(1, ia.modules, cell(increase,1));
            
            
            old_connections = ia.connections;
            ia.connections = zeros(new_size, new_size);
            ia.connections(1:curr_size, 1:curr_size) = old_connections;
            
            ia.modules_type = cat(1, ia.modules_type, zeros(increase,1));
            
            ia.graphs = cat(1, ia.graphs, cell(increase,1));
        end
        
    end
end

