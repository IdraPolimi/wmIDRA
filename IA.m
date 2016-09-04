classdef IA < handle
    
    properties 
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
            ia.ExplorationPercentage = 0.05;
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
            module.output_size = input_size;
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
            
            module.categorized_input = zeros(1, module.n_centroids);
            
            module.update = false;
            module.reward = 0;
            
            module.time = 0;
            
            ia.modules{index} = module;
            ia.UpdateGraph();
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
%             ia.UpdateGraph();
        end
        
        function Train(ia)
            ia.UpdateInputModules();
            ia.TrainIntentionalModules();
        end
        
        function ClearInputOutput(ia)
            for index = 1:ia.CountModules()
                ia.modules{index}.input = [];
                ia.modules{index}.output = [];
                ia.modules{index}.input_changed = true;
            end
        end
    end
    
    methods
        
        function size = GetModuleOutputSize(ia, indeces)
            size = 0;
            for ii = 1:length(indeces)
                index = indeces(ii);
                size = max(size, length(ia.modules{index}.output));
            end
        end
        
        function size = GetModuleInputSize(ia, indeces)
            size = 0;
            for ii = 1:length(indeces)
                index = indeces(ii);
                size = max(size, length(ia.modules{index}.input));
            end
        end
        
        function out = GetModuleOutput(ia, indeces)
            out = [];
            
            for ii = 1:length(indeces)
                index = indeces(ii);
                out = cat(1, out, ia.modules{index}.output);
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
                ia.wm.SetLearningRate(module.wm_index, ia.LearningRate);
                ia.wm.SetExplorationPercentage(module.wm_index, ia.ExplorationPercentage*2);

                
                
                update = module.input_changed || module.time > 20;

                if  update
                    module.time = 0;

                    ia.wm.SetReward(module.wm_index, module.reward);
                    module.reward = 0;
                    
                    state = module.categorized_input;
                    
                    ia.wm.SetState(module.wm_index, state);
                    
                    patch_pool = unique( cat(1, module.input, module.output), 'rows' );
                    
                    [~, ~, ~, chunks] = ia.ProcessForward(module, patch_pool);
                    
                    if isempty(chunks)
                        chunks = zeros(1, module.n_centroids);
                    end

                    ia.wm.SetCandidateChunks(module.wm_index, chunks);
                    ia.wm.EpisodeTick(module.wm_index);

                    retained_chunks = ia.wm.GetRetainedChunks(module.wm_index);
                    
                    module.output_changed = false;
                    if isempty(retained_chunks)
                        module.output_changed = ~isempty(module.output);
                        module.output =  [];
                    else
                        out = patch_pool(ismember(chunks, retained_chunks, 'rows'), :);
                        
                        if size(out, 1) ~= size(module.output, 1)
                            module.output_changed = true;
                        else
                            module.output_changed = false;%any(out ~= module.output);
                        end
                        module.output = unique( out, 'rows' );
                    end
                end
                ia.modules{index} = module;
            end
        end
        
        function UpdateOutputModules(ia)
            indeces = find(ia.modules_type == ia.TYPE_OUT);
            
            for ii = 1:length(indeces)
                index = indeces(ii);
                module = ia.modules{index};
                
                update = module.input_changed && any(module.categorized_input > 0);
                module.update = false;
                
                if  update
                    module.time = 0;
                    
                    
                    ia.wm.SetReward(module.wm_index, module.reward);
                    module.reward = 0;
                    
                    state = module.categorized_input;
                    
                    ia.wm.SetState(module.wm_index, 1);
                    

                    
                    chunks = zeros(module.n_actions, module.n_centroids * module.n_actions);
                    
                    for kk = 1:module.n_centroids
                        cols = ((kk - 1) * module.n_actions + 1):(module.n_actions*kk);
                        chunks(:, cols) = eye(module.n_actions) * module.similarities(kk);
                    end
                    
                    
                    ia.wm.SetCandidateChunks(module.wm_index, chunks);
                    ia.wm.EpisodeTick(module.wm_index);
                        
                    retained_chunks = ia.wm.GetRetainedChunks(module.wm_index);
                    
                    retained_chunks(retained_chunks > 0) = 1;
                    
                    module.output_changed = false;
                    
                    out = zeros(1, module.n_actions);
                    if ~isempty(retained_chunks)
                        max_s = find(retained_chunks);
                        mod_s = mod(max_s, module.n_actions);
                        max_s = max_s - mod_s;
                        max_s = max_s / module.n_actions + double(mod_s > 0);
                        cols = ((max_s - 1) * module.n_actions + 1):(module.n_actions*max_s);
                        out = sum(retained_chunks(:,cols), 1);
                    end
                    module.output_changed = true;
                    module.output = out;
                end
                
                if isempty(module.output)
                    module.output = zeros(1, module.n_actions);
                end
                ia.modules{index} = module;
            end
        end
        
        function UpdateIntentionalModules(ia)
                
            indeces = find(ia.modules_type == ia.TYPE_IM);
            indeces = cat(1, indeces, find(ia.modules_type == ia.TYPE_OUT));
            
            for ii = 1:length(indeces)
                index = indeces(ii);
                module = ia.modules{index};
                
                input = ia.GetModuleOutput(module.sources);
                
                if isempty(input) && ~isempty(module.input) || ~isempty(input) && isempty(module.input)
                    module.input_changed = true;
                end
                    
                if isempty(input)
                    continue;
                end
                
                module.input = input;
                no_input = isempty(input);
                
                [module.ica_weights, module.in_pca, sim, sim_max] = ia.ProcessForward(module, input);
                
                if size(sim, 1) ~= size(module.similarities, 1)
                    module.input_changed = true;
                else
                    module.input_changed = any(module.categorized_input ~= sum(sim_max, 1));
                end
                
                module.similarities = max(sim, [], 1);
                
                if no_input
                    module.categorized_input = zeros(1, module.n_centroids);
                    module.activation = 0;
                else
                    module.categorized_input = sum(sim_max, 1);
                    module.activation = min(max(module.similarities, [], 2));
                end
                
                
                ia.wm.SetGamma(module.wm_index, 0.95);
                
                
%                 plotted = module.ica_weights;
%                 plotted_size = size(plotted);
%                 additions = (plotted_size(2)+1):3;
%                 plotted(:, additions) = zeros(plotted_size(1), length(additions));
%                 
%                 ia.graphs{index}.output.XData = plotted(:, 1);
%                 ia.graphs{index}.output.YData = plotted(:, 2);
%                 ia.graphs{index}.output.ZData = plotted(:, 3);

                module.time = module.time + 1;
                ia.modules{index} = module;
                
            end
            
        end
        
        function training_complete = TrainIntentionalModules(ia)
            indeces = find(ia.modules_type == ia.TYPE_BOOTSTRAPING);
            
            training_complete = 1;
            
            for ii = 1:length(indeces)
                index = indeces(ii);
                
                sources = ia.GetParentModules(index);
                ia.modules{index}.input = ia.GetModuleOutput(sources);
                ia.modules{index}.output = ia.modules{index}.input;
                
                training_complete = training_complete * ia.TrainIntentionalModule(index);
            end
        end
        
        function full = UpdateTrainingSet(ia, index, input)
            
           module = ia.modules{index};
           curr_count = module.training_count;
           max_count = size(module.training_set, 1);
           input_length = size(input, 1);
           
           new_count = min(curr_count + input_length, max_count);
           
           if curr_count < max_count
               module.training_set(curr_count + 1:new_count, :) = input(1:new_count-curr_count, :);
               module.training_count = new_count;
           end
           
           full = logical(new_count >= max_count);
           
           ia.modules{index} = module;
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
                
                ia.wm.SetState(module.wm_index, module.wm_init_state);
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
        
        function training_complete = TrainIntentionalModule(ia, index)
            module = ia.modules{index};
            
            input = module.input;
            
            training_complete = false;
            
            if ia.UpdateTrainingSet(index, input)
                training_complete = true;
                training_set = ia.modules{index}.training_set;
                
                %---------------- MODULE TRAINING ------------------ begin
                sigin = training_set;
                
                %PCA
                [pc.coeff, pc.score, pc.latent, pc.tsquared, pc.explained, pc.mu] = pca(sigin);
                
                perc = cumsum(pc.latent) / sum(pc.latent);
                perc(perc > 1) = 0;
                perc(perc > 0 ) = 1;
                perc( 1:3 ) = 1;
                k = sum(perc);

                [pc.coeff, pc.score, pc.latent, pc.tsquared, pc.explained, pc.mu] = pca(sigin, 'NumComponents', k);
                
                pc.inv_coeff = pinv(pc.coeff');
                %ICA
                
                [ica.ics, ica.A, ica.W] = fastica(pc.score');
                
                
                ica.n_ics = size(ica.ics, 1);
                
                ica.mean_ics = mean(abs(ica.ics), 2)';
                
                %--
                module.ica = ica;
                module.pca = pc;
                %--
                
                [ica_weights, ~, ~, ~] = ia.ProcessForward(module, training_set);
                
                ica_weights(all(ica_weights==0, 2), :) = [];
                %--------
                
                
                
                som_w = min(16, module.n_centroids);
                som_h = ceil(module.n_centroids / som_w);
                
                som = selforgmap([som_w som_h], 100, 3, 'hextop','linkdist');
                
                module.n_centroids = som_h * som_w;
                
                som = train(som, ica_weights);
                
                module.som = som;
                
                
                [~, ~, ~, classes] = ia.ProcessForward(module, training_set);
                
                ids = vec2ind(classes');
                
%                 [ids, centroids, ~, ~] = kmeans(ica_weights, module.n_centroids, 'EmptyAction','singleton', 'Distance', 'cosine','MaxIter',500);
%                 
%                 module.centroids = centroids;
                %------
                if module.output_module
                    module.wm_init_state = 1;
                    module.wm_index = ia.wm.NewWMModule(module.n_retained_chunks, 1, module.n_centroids * module.n_actions);
                    module.out_mask = zeros(1, module.n_actions);
                else
                    module.wm_init_state = zeros(1, module.n_centroids);
                    module.wm_index = ia.wm.NewWMModule(module.n_retained_chunks, module.n_centroids, module.n_centroids);
                    module.out_mask = zeros(1, ica.n_ics);
                end
                
                
                if module.output_module
                    ia.modules_type(index) = ia.TYPE_OUT;
                else
                    ia.modules_type(index) = ia.TYPE_IM;
                end
                
                module.bootstraping = false;
                ia.modules{index} = module;
                %---------------- MODULE TRAINING ------------------ end
                
            
            
                module = ia.modules{index};
                
%                 [current_output, ~, ~, ~] = ia.ProcessForward(module, input(end,:));
                figure(index * 13);
                
                hold on;
                
                plotted = ica.ics';
                
                for aa = size(plotted,2)+1:3
                    plotted(:,aa) =  zeros(size(plotted,1),1);
                end
%                 for aa = size(centroids,2)+1:3
%                     centroids(:,aa) = zeros(size(centroids,1),1);
%                 end
%                 for aa = size(current_output,2)+1:3
%                     current_output(:,aa) = zeros(size(current_output,1),1);
%                 end
                
                
                ids_unique = unique(ids);
                
                for cc = 1:length(ids_unique)
                    indeces = find(ids == ids_unique(cc));
                    scatter3(plotted(indeces,1), plotted(indeces,2), plotted(indeces,3), '.');
                end
                grid on;
%                 ia.graphs{index}.centroids = scatter3(centroids(:,1 ),centroids(:, 2),centroids(:, 3), 'B');
%                 ia.graphs{index}.output = scatter3(current_output(:,1 ),current_output(:, 2),current_output(:, 3), 'R');
                hold off;
                axis auto;
                grid on;
                
            end
        end
        
        function [ica_weights, in_pca, sim, sim_max] = ProcessForward(ia, module, inputs)
            
            ica_weights = [];
            in_pca = [];
            sim = [];
            sim_max = [];
            if isempty(inputs)
                return;
            end
            
            ica = module.ica;
            pcs = module.pca;
            mu = pcs.mu;
            inv_coeff = pcs.inv_coeff;
            
            in_pca = (inputs - repmat(mu, size(inputs,1), 1)) * inv_coeff;
            
            ica_weights = (ica.W * in_pca');
            %---
            mean_ics = repmat(ica.mean_ics, size(inputs,1), 1)';
            ica_weights(abs(ica_weights) < 0.25 * mean_ics) = 0;
            ica_weights( :, ~any(ica_weights, 1) ) = [];
            if ~isfield(module, 'som')
                return;
            end
            
            net = module.som;
            sim = net(ica_weights)';
            
            sim_max = sim;
            %---
%             mean_ics = repmat(ica.mean_ics, size(inputs,1), 1);
%             
%             ica_weights(abs(ica_weights) < 0.2 * mean_ics) = 0;
%             
%             sim = ia.GetCentroidsDistance(module, ica_weights);
%             
%             sim2 = sim;
%             sim(sim < ia.threshold) = 0;
%             sim_max = double(sim2 == repmat(max(sim2, [], 2), 1, size(sim2, 2)));
%             sim_max = sum(sim_max, 1);
            %---
            
%             sim2(sim2 < ia.threshold) = 0;
%             sim2( ~any(sim2,2), : ) = [];
%             if isempty(sim2)
%                 sim_max = zeros(1, module.n_centroids);
%             else
%                 sim_max = double(sim2 == repmat(max(sim2, [], 2), 1, size(sim2, 2)));
%             end
        end
       
        function sim = GetCentroidsDistance(~, module, points)
            centroids = module.centroids;
            
            count_centroids = size(centroids, 1);
            count_points = size(points, 1);
            
            dists = zeros(count_points, count_centroids);
            
            for ii = 1:count_points
                point = points(ii,:);
                
                dd = pdist(cat(1, point, centroids), 'cosine');

                dd = dd(1:count_centroids);

                dd = reshape(dd, count_centroids, 1)';

                dists(ii,:) = dd;
            end
            
            dists(dists > 1) = 1;
            sim = 1 - abs(dists);

        end
        
        function index = NextIndex(ia, nodeType)
            while ia.CountModules() >= ia.MaxSize()
                ia.DoubleSize();
            end
            
            index = ia.CountModules() + 1;
            ia.next_index = ia.next_index + 1;
            
            ia.modules_type(index) = nodeType;
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

