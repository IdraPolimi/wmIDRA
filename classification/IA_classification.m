classdef IA_classification < handle
    %IA Summary of this class goes here
    %   Detailed explanation goes here
    
    
    properties ( Access = private )
       TYPE_INPUT = 1
       TYPE_CONTEXT = 2
       TYPE_IM = 3
       TYPE_BOOTSTRAPING = 4
       TYPE_EMPTY = 0
    end
    
    
    properties 
        context
        modules
        modules_type
        next_index
        connections
        activations
        
        threshold
        graphs
        
        gain
        ff
        
        layers_count
        layer_modules_count
        
        sampling_percentage
        
        network_graph
        network_graph_plot
        network_graph_count
        network_graph_fig_no
    end
    
    properties
        UseGPU
        im_ca
    end
    
    methods
        
        function ia = IA_classification(treshold, gain)
            size = 1;
            
            ia.UseGPU = 0;
            
            ia.modules = cell(size,1);
            ia.next_index = 1;
            ia.modules_type = zeros(size,1);
            ia.connections = zeros(size, 1);
            ia.activations = zeros(size, 1);
            ia.threshold = treshold;
            ia.gain = gain;
            
            ia.ff = 1;
            
            ia.layers_count = 5;
            ia.layer_modules_count = 3;
            
            ia.sampling_percentage = 1;
            
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
           r = ia.UseGPU();
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
            
            module.cache_size = 1000;
            module.cache_count = 0;
            module.cache = zeros(module.cache_size, input_size);
            
            ia.modules{index} = module;
            input_module = FilterNode(ia, index, input_size, filterFunction);
        end
        
        function bs = IsBootstraping(ia, indeces)
            bs = ia.modules_type(indeces);
            
            bs = bs == ia.TYPE_BOOTSTRAPING;
        end
        
        function index = NewIntentionalModule(ia, sources, training_set_size, output_size)
            index = ia.NextIndex(ia.TYPE_BOOTSTRAPING);
            ia.connections(index, sources) = 1;
            input_size = ia.GetModuleOutputSize(sources);
            
            module = [];
            module.index = index;
            module.sources = sources;
            module.input = zeros( 1, input_size );
            module.output = zeros( 1, output_size );
            module.bootstraping = true;
            module.plot_activation = 0;
            module.sampling_weights = ones(training_set_size,1);
            module.training_count = 0;
            module.training_max_count = training_set_size;
            module.ica = [];
            module.pca = [];
            module.centroids = [];
            module.centroids_mean_distance = [];
            module.activation = 0;
            
            module.W = eye(output_size);
            module.W0 = zeros(1, output_size);
            
            ia.modules{index} = module;
            
        end
        
        function Update(ia)
            ia.UpdateInputModules();
            ia.UpdateIntentionalModules();
        	ia.UpdateContextModules();
            ia.UpdateActivations();
            ia.UpdateGraph();
        end
        
    end
    
    methods
        
        function ss = GetModuleOutputSize(ia, indeces)
            ss = 0;
            for ii = 1:length(indeces)
                index = indeces(ii);
                ss = ss + size(ia.modules{index}.output, 2);
            end
        end
        
        function ss = GetModuleInputSize(ia, indeces)
            ss = 0;
            for ii = 1:length(indeces)
                index = indeces(ii);
                ss = ss + size(ia.modules{index}.input, 2);
            end
        end
        
        function out = GetModuleOutput(ia, indeces)
            out = [];
            for ii = 1:length(indeces)
                index = indeces(ii);
                out = cat(2, out, ia.modules{index}.output);
            end
        end
        
        function ss = GetModuleSimilaritiesSize(ia, indeces)
            ss = length(indeces);
        end
        
        function out = GetModuleSimilarities(ia, indeces)
            out = [];
            for ii = 1:length(indeces)
                index = indeces(ii);
                out = cat(2, out, ia.modules{index}.similarities);
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
            
            booting = false;
            for ii = 1:length(childrens)
               index = childrens(ii);
               booting = booting || ia.modules{index}.bootstraping;
            end
        end
        
        function UpdateInputModules(ia)
            indeces = find(ia.modules_type == ia.TYPE_INPUT);
            
            for ii = 1:length(indeces)
                index = indeces(ii);
                ia.modules{index}.output = ia.modules{index}.input;
                
            end
        end
        
        function UpdateIntentionalModules(ia)
            
            
            indeces = find(ia.modules_type == ia.TYPE_IM);
            
            for ii = 1:length(indeces)
                index = indeces(ii);
                module = ia.modules{index};
                
                input = ia.GetModuleOutput(module.sources);
                ia.modules{index}.input = input;
                output = ia.ProcessForward(module,input);
                
                
                sim = ia.GetCentroidsDistance(module, ia.gain, output);
%                 sim = max(sim, [], 2);
                
                
                
                sim = max(sim, [], 2);
%                 
                ia.modules{index}.similarities = sim;
            	ia.modules{index}.output = output;%repmat(sim, 1, size(input,2)) .* input;
                
                ia.modules{index}.activation = max(max(sim));
            end
            
            indeces = find(ia.modules_type == ia.TYPE_BOOTSTRAPING);
            
            trained = false;
            for ii = 1:length(indeces)
                index = indeces(ii);
                trained = trained || ia.TrainIntentionalModule(index);
            end
            
            indeces = find(ia.modules_type == ia.TYPE_IM);
            if ~trained
                for ii = 1:length(indeces)
                    index = indeces(ii);
                    
                    siblings = ia.GetSiblingModules(index);
                    
                    sim = ia.modules{index}.similarities;
                    
                    sim = sim ./ sum([sim, ia.GetModuleSimilarities(siblings)] ,2);
                    ia.modules{index}.activation = max(max(sim));
                end
            end
            
        end
        
        function UpdateContextModules(ia)
            indeces = find(ia.modules_type == ia.TYPE_CONTEXT);
            
            if isempty(indeces)
                return;
            end
            
            for ii = 1:length(indeces)
                index = indeces(ii);
                module = ia.modules{index};
                
                src = module.source;
                input = ia.GetModuleOutput(src);
                ia.context.SetActivations(module.receptors, input);
            end
            
            ia.context.Update();
            
            for ii = 1:length(indeces)
                index = indeces(ii);
                receptors = ia.modules{index}.receptors;
                a = ia.context.GetActivations(receptors);
                ia.modules{index}.output = a;
                ia.modules{index}.activation = max(a);
            end
        end
        
        function UpdateActivations(ia)
            indeces = cat(1, find(ia.modules_type == ia.TYPE_IM), find(ia.modules_type == ia.TYPE_CONTEXT));
            
            for ii = 1:length(indeces)
                index = indeces(ii);
                
                ia.activations(index) = ia.modules{index}.activation;
            end
            
            ia.activations(ia.modules_type == ia.TYPE_INPUT) = 1;
        end
        
        function full = UpdateTrainingSet(ia, index, input)
           module = ia.modules{index};
           curr_count = module.training_count;
           max_count = module.training_max_count;
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
            
            for ii = 1:count
                pp = ia.modules{ii}.activation;
                pp = max(min(pp,1),0);
                
                highlight(ia.network_graph_plot, ii, 'NodeColor', [1 - pp, pp, 0]);
            end
            highlight(ia.network_graph_plot,ia.network_graph,'EdgeColor',[0 0.4470 0.7410],'LineWidth',0.33);

%             max_active_adj = ia.connections(1:count, 1:count);
%             h = digraph(max_active_adj);
%             highlight(ia.network_graph_plot, h, 'EdgeColor','black','LineWidth',2)
        end
        
        function a = GetNodesActivation(ia, indeces)
            if isempty(indeces)
                a = 0;
            else
                a = ia.activations(indeces);
            end
        end
        
        function trained = TrainIntentionalModule(ia, index)
            module = ia.modules{index};
            
            sources = module.sources;
            
            sa = ia.GetNodesActivation(sources);
            
            
            if any(ia.IsBootstraping(sources))
                return;
            end
            
%             if max(sa) < ia.threshold
%                 return;
%             end
            
            input = ia.GetModuleOutput(sources);
            
            trained = false;
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
                
                [pc.coeff, pc.score, pc.latent, pc.tsquared, pc.explained, pc.mu] = pca(sigin);
                
                
                
                perc = cumsum(pc.latent) / sum(pc.latent);
                
                hold_var = 1;
                perc(perc > hold_var) = 0;
                
                
                perc(perc > 0) = 1;
                
                pc.coeff =  pc.coeff(:, logical(perc));
                
                pc.score = (sigin - repmat(pc.mu, size(sigin,1), 1)) * pinv(pc.coeff');
                
                [ica.ics, ica.A, ica.W] = fastica(pc.score', 'maxNumIterations', 1000);
                module.ica = ica;
                module.pca = pc;
                
                k = ia.GetModuleOutputSize(index);
                
                tset_W = ica.ics';%ia.ProcessForward(module, training_set);
                
                
                [centroids, ~, meand, covmat] = categorizeInput(tset_W, k);
                
                [ ia.modules{index}.w_mean, ia.modules{index}.w_variance ] = naiveClassification( tset_W );
                
                ia.modules{index}.centroids = centroids;
                ia.modules{index}.covmat = covmat;
                ia.modules{index}.centroids_mean_distance = meand';
                ia.modules{index}.ica = ica;
                ia.modules{index}.pca = pc;
                ia.modules{index}.bootstraping = false;
                ia.modules_type(index) = ia.TYPE_IM;
                
                trained = true;
                
                %---------------- MODULE TRAINING ------------------ end
                
%                 module = ia.modules{index};
%                 current_output = ia.ProcessForward(module, input);
%                 figure(index * 10);
%                 
%                 hold on;
%                 tset_plot = ia.ProcessForward(module, sampled_training_set);
%                 
%                 for aa = size(tset_plot,2)+1:3
%                     tset_plot(:,aa) = zeros(size(tset_plot,1),1);
%                 end
%                 for aa = size(centroids,2)+1:3
%                     centroids(:,aa) = zeros(size(centroids,1),1);
%                 end
%                 for aa = size(current_output,2)+1:3
%                     current_output(:,aa) = zeros(size(current_output,1),1);
%                 end
%                 ia.graphs{index}.training = scatter3(tset_plot(:,1 ),tset_plot(:, 2),tset_plot(:, 3), 'G','.');
%                 ia.graphs{index}.centroids = scatter3(centroids(:,1 ),centroids(:, 2),centroids(:, 3), 'B');
%                 ia.graphs{index}.output = scatter3(current_output(:,1 ),current_output(:, 2),current_output(:, 3), 'R','.');
%                 hold off;
%                 axis manual;
%                 grid on;
                
            end
        end
        
        function inputs = ProcessBackward(~, module, points)
            
            ics = module.ica.ics;
            
            pcs = module.pca;
            mu = pcs.mu;
            coeff = pcs.coeff';
            
            
            
            inputs = points * ics;
            
            inputs = (inputs * coeff) + repmat(mu, size(inputs,1), 1);
        end
        
        function outputs = ProcessForward(~, module, inputs)
            
            ics = module.ica.ics;
            pcs = module.pca;
            mu = pcs.mu;
            coeff = pcs.coeff';
            
            a = (inputs - repmat(mu, size(inputs,1), 1));
            
            outputs = (module.ica.W * a')';
        end
       
        function sim = GetCentroidsDistance(ia, module, gain, points)
            centroids = module.centroids;
            
            covmat = module.covmat;
            
            count_centroids = size(centroids, 1);
            count_points = size(points, 1);
            
            pds = zeros(count_points, count_centroids);
            
            for ii = 1:count_centroids
               pds(:,ii) = mvnpdf(points,centroids(ii,:),covmat(:,:,ii));
            end
            
            
            
            
            sim = pds;
            
            
            w_mean = repmat(module.w_mean, count_points, 1);
            w_var = repmat(module.w_variance, count_points, 1);
            
            sim = exp(-(points - w_mean).^2 ./ (2 * w_var.^2) );
            
            sim = prod(sim, 2) .* gain;
            
        end
        
        function [centroid, centroid_index, dist] = FindNearestCentroid(ia, module, point)
            centroids = module.centroids;
            dd = ia.GetCentroidsDistance(centroids, ia.gain, point);
            
            dist = min(dd);
            
            centroid_index = find(dd == dist);
            centroid_index = centroid_index(1);
            
            centroid = centroids(centroid_index, :);
            
            centroid_index = find(centroid_index);
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
            ia.activations = cat(1, ia.activations, zeros(increase,1));
            
            ia.graphs = cat(1, ia.graphs, cell(increase,1));
        end
        
    end
end

