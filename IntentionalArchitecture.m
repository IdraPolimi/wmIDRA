classdef IntentionalArchitecture < handle
    %IDRA INTENTIONALARCHITECTURE
    
    properties
       TYPE_INPUT = 1
       TYPE_CONTEXT = 2
       TYPE_IM = 3
       TYPE_EMPTY = 0
    end
    
    properties
        im_connections          % N x N adjacency matrix containing the connections between intentional 
                                % modules
              
        im_activations
        im_bootstraping         % N x 1 array determining whether an intentional module is bootstrabing or not
        
        
        im_type
        
        
        
        im_ca                   % N x OUT matrix containing categories activation
        
        im_input_nodes
        
        im_count
        
        im_centroids
        
        im_gathered_input
        im_gathered_input_count
        
        im_ica
        im_ica_size
        im_pca
        
        im_resonating
        
        modules
    end
    
    properties
       context
       receptors_source
       receptors_indeces
    end
    
    properties
        ICASize
        NodesInputSize
        NodesOutputSize
        CountNodes
        MaxSize
        UseGPU
        TrainingSetSize
    end
    
    methods
        function ia = IntentionalArchitecture(k, categoriesPerModule, icaSize, trainingSetSize)
            maxSize = 1;
            ia.im_input_nodes = k;
            
            output_size = categoriesPerModule;
            input_size = k * output_size;
            
            ia.modules = cell(1,maxSize);
            
            ia.context = SOM_Context(20, output_size * maxSize, 5);
            
            ia.receptors_source = zeros(maxSize, 1);
            ia.receptors_indeces = zeros(maxSize, output_size);
            
            ia.im_connections = zeros(maxSize, maxSize);
            
            ia.im_bootstraping = zeros(maxSize, 1);
            
            ia.im_activations = zeros(maxSize, 1);
            
            ia.im_type = ones(maxSize, 1) * ia.TYPE_EMPTY;
            
        
            ia.im_ca = zeros(maxSize, output_size);
            
            ia.im_ca(1, :) = ones(output_size, 1);
            ia.im_count = 2;
            
            ia.im_gathered_input = zeros(maxSize, input_size,trainingSetSize);
            ia.im_gathered_input_count = ones(maxSize,1);
            ia.im_centroids = zeros(maxSize, icaSize, output_size);
            
            ia.UseGPU = 0;
            
            ia.im_ica = cell(maxSize, 1);
            ia.im_ica_size = icaSize;
            ia.im_pca = cell(maxSize, 1); 
            ia.im_resonating = zeros(maxSize,1);
        end
        
        function s = get.ICASize(ia)
            s = ia.im_ica_size;
        end
        
        function s = get.TrainingSetSize(ia)
            s = size(ia.im_gathered_input,3);
        end
        
        function s = get.NodesInputSize(ia)
            s = ia.im_input_nodes * size(ia.im_ca, 2);
        end
        
        function s = get.NodesOutputSize(ia)
           s = size(ia.im_ca, 2); 
        end
        
        function size = get.MaxSize(ia)
            size = length(ia.im_bootstraping);
        end
        
        function count = get.CountNodes(ia)
            count = ia.im_count;
        end
        
        function res = get.UseGPU(ia)
           res = ia.UseGPU ~= 0;
        end
        
        function set.UseGPU(ia, val)
           ia.UseGPU = val;
           r = ia.UseGPU();
           ia.context.UseGPU = r;
        end
       
        function Update(ia)
            ia.UpdateInputNodes();
            ia.UpdateIntentionalNodes();
            ia.UpdateContextNodes();
            ia.UpdateNodesActivation();
        end
        
    end
    
    methods (Access = public)
        %Node creation methods
        
        function node = NewFilterNode(ia, filterFunction)
            
            index = ia.NextIndex(ia.TYPE_INPUT);
            
            if index < 1
                node = 0;
                return;
            end
            
            node = FilterNode(ia, index, ia.NodesOutputSize(), filterFunction);
            
        end
        
        function node = NewIntentionalModule(ia, inputs)
            inputs = unique(inputs);
            k = ia.NodesInputSize() / ia.NodesOutputSize();
            input_length = length(inputs);
            
            if input_length < k
                inputs = [inputs, ones(1, k - input_length) ];
            end
            
            inputs = inputs(1:k);
            
            index = ia.NextIndex(ia.TYPE_IM);
            
            if index < 1
                node = 0;
                return;
            end
            
            node = IntentionalModule(ia, index);
            
            ia.SetBootstraping(index, 1);
            
            ia.im_connections(index, inputs) = 1;
            
            ia.NewContextNode(index);
        end
        
    end
    
    
    methods (Access = public)
        % Methods used by the nodes to access their data
        
        function bs = IsBootstraping(ia, indeces)
            bs = ia.im_bootstraping(indeces);
        end
        
        function SetBootstraping(ia, indeces, values)
            ia.im_bootstraping(indeces) = values;
        end
        
        
        function a = GetNodesActivation(ia, indeces)
        	a = ia.im_activations(indeces);
        end
        
        % Setter for the activations of the nodes' categories.
        % InputNode:        categories activations are it's input
        % IntentionalNode : categories activations are it's output categories
        % ContextNode:      categories activations are the values of the context's
        % som receptor
        function SetCategoriesActivation(ia, indeces, values)
            ia.im_ca(indeces,:) = values;
        end
        
        function SetModuleInput(ia, index, values)
           ia.SetCategoriesActivation(index, values);
        end
        
    end
    
    methods (Access = private)
        
        function NewContextNode(ia, im_index)
            
            index = ia.NextIndex(ia.TYPE_CONTEXT);
            
            if index < 1
                return;
            end
            
            ia.receptors_source(index) = im_index;
            
            ia.receptors_indeces(index, :) = ia.context.NextAvailableIndeces(ia.NodesOutputSize());
            
        end
        
            
        
        % Allocates and returns the first available index for a new
        % node. A return value of -1 means that the intentional
        % architecture is full.
        function index = NextIndex(ia, nodeType)
            while ia.CountNodes() >= ia.MaxSize()
                ia.DoubleSize();
            end
            
            index = ia.CountNodes();
            ia.im_count = ia.im_count + 1;
            
            ia.im_type(index) = nodeType;
        end
        
        function UpdateInputNodes(~)
            %indeces = ia.im_type(:) == ia.TYPE_INPUT;
            % ia.im_ca(index,:) = rand(activations_size, 1);
        end
        
        function UpdateIntentionalNodes(ia)
            
            type = ia.im_type;
            ca = ia.im_ca;
            conn = ia.im_connections;
            indeces = find(type == ia.TYPE_IM);
            
            
            k = ia.NodesInputSize() / ia.NodesOutputSize();
            
            for ii = 1:length(indeces)
                index = indeces(ii);
                module = IntentionalModule(ia, index);
                in_nodes = find(conn(index,:) == 1);
            	in = ca(in_nodes,:);
                kk = size(in, 1) - k;
                in = reshape(in', 1, size(in,1) * size(in,2))';
                in = cat(1, in, ones(ia.NodesInputSize() * kk, 1));
                
                
                if ia.IsBootstraping(index)
                    % If the module is bootstraping
                    
                    if ia.im_gathered_input_count(index) > size(ia.im_gathered_input,3)
                        % If we have enough samples we perform ica
                        
                        gathered_input = ia.im_gathered_input(index,:,:);
                        
                        sigin = reshape(gathered_input, ia.NodesInputSize(), size(gathered_input,3))';
                        
                        sigin = sigin(2:end-1, :);
                        
                        [ic.ics, ic.A, ic.W] = fastica(sigin, 'numOfIc', ia.ICASize());
                        
                        A = sigin * pinv(ic.ics);
                        
                        ia.im_ica{index} = ic;
                        ia.SetBootstraping(index,0);
                        [pc.coeff, pc.score, pc.latent, pc.tsquared, pc.explained, pc.mu] = pca(A);
                        
                        updatedCategories = categorizeInput(pc.score',ia.NodesOutputSize());
                        ia.im_centroids(index,:,:) = updatedCategories;
                        
                        ia.im_pca{index} = pc;
                        
%                         ia.NewIntentionalModule(in_nodes);
%                         ia.NewIntentionalModule(index);
                        
                    else
                        % Otherwise we accumulate the sample if input
                        % modules are activated
                        parents = in_nodes;
                        in_activations = ia.GetNodesActivation(parents);
                        
                        siblings = find(any(ia.im_connections(:,in_nodes), 2) == true);
%                        
                        siblings_activations = ia.GetNodesActivation(siblings);
                        
                        if (min(in_activations) > 0.66) && (max(siblings_activations) < 0.33);
                             ia.im_gathered_input(index, :, ia.im_gathered_input_count(index)) = in;
                            ia.im_gathered_input_count(index) = ia.im_gathered_input_count(index) + 1;
                        end
                    end
                else
                    % If the module is not bootstraping we process the
                    % sample through kmeans
                    ia.im_gathered_input(index,:, 1:end-1) = ia.im_gathered_input(index,:, 2:end);
                    ia.im_gathered_input(index,:, end) = in;
                    
                    if index == 8
                        a=1;
                    end
                    
                    
                    in_pc=module.ProcessForward(in');
                    
                    dist = module.GetCentroidsDistance(in_pc);
                    
                    [centroid, centroid_index, resonating] = module.IsResonating();
                   
                    ia.im_resonating(index) = resonating;
                    
                    if ia.GetNodesActivation(index) > 0.5
                        a = module.Activation*0.1;
                        if resonating
                            ia.im_centroids(index,:,centroid_index) = (1 - a) * centroid + a * in_pc;
                        end
                    end
                        
                   
                   ia.SetCategoriesActivation(index, dist);
                   
                   %--------------------------------
                   if index == 4 || index == 8
                       score = ia.im_pca{index}.score;

                       figure(index*10);
                       
                       scatter3(score(:,1 ),score(:, 2),score(:, 3), 'G');
                       %axis([-0.01 0.01 -0.01 0.01 -0.01 0.01]);
                       hold on;
                       scatter3(in_pc(1),in_pc(2),in_pc(3),'R');
                       scatter3(ia.im_centroids(index,1,:),ia.im_centroids(index,2,:),ia.im_centroids(index,3,:),'B');
                       
                       hold off;
                   end
                   %--------------------------------
                end
                
            end
            
        end
        
        function UpdateContextNodes(ia)
            
            indeces = ia.im_type(:) == ia.TYPE_CONTEXT;
            
            idx = ia.receptors_indeces(indeces, :);
            r_indeces = idx;
            
            
            receptors_input = ia.im_ca(ia.receptors_source(indeces), :);
            ia.context.SetActivations(r_indeces, receptors_input);
            
            ia.context.Update();
            
            node_activations = ia.context.GetActivations(r_indeces);
            
            ia.im_ca(indeces, :) = node_activations;
            
        end
        
        function UpdateNodesActivation(ia)
            
            mask = ia.im_type == ia.TYPE_INPUT;
            ia.im_activations(mask) = 1;
            
            mask = ia.im_type == ia.TYPE_IM;
            ia.im_activations(mask) = max(ia.im_ca(mask), [], 2);
            
            mask = ia.im_type == ia.TYPE_CONTEXT;
            ia.im_activations(mask) = max(ia.im_ca(mask), [], 2);
            
            mask = ia.im_type == ia.TYPE_EMPTY;
            ia.im_activations(mask) = 1;
        end
        
        function DoubleSize(ia)
            
            curr_size = ia.MaxSize();
            new_size = 2 * curr_size;
            
            inc_size = new_size - curr_size;
            
            old = ia.im_connections;
            
            ia.im_connections = zeros(new_size, new_size);
            ia.im_connections(1:curr_size, 1:curr_size) = old;
                                
            ia.im_activations = cat(1, ia.im_activations, zeros(inc_size, 1));
            ia.im_bootstraping = cat(1, ia.im_bootstraping, zeros(inc_size, 1));
            
            ia.im_type = cat(1, ia.im_type, ones(inc_size, 1) * ia.TYPE_EMPTY);
            

            ia.im_ca = cat(1, ia.im_ca, zeros(inc_size, ia.NodesOutputSize()));

            ia.receptors_source = cat(1, ia.receptors_source, zeros(inc_size, 1));
            
            ia.receptors_indeces = cat(1, ia.receptors_indeces, zeros(inc_size, size(ia.receptors_indeces,2)));
            
            
            
            ia.im_gathered_input = cat(1,ia.im_gathered_input, zeros(inc_size, ia.NodesInputSize(),ia.TrainingSetSize()));
            ia.im_gathered_input_count = cat(1, ia.im_gathered_input_count, ones(inc_size,1));
            ia.im_centroids = cat(1, ia.im_centroids, zeros(inc_size, ia.ICASize(), ia.NodesOutputSize()));
           
            ia.im_pca = cat(1, ia.im_pca, cell(inc_size, 1));
            ia.im_ica = cat(1, ia.im_ica, cell(inc_size, 1));
            
            ia.modules = cat(1, ia.modules, cell(inc_size, 1));
            
            ia.im_resonating = cat(1, ia.im_resonating, zeros(inc_size, 1)); 
            ia.context.DoubleSize();
        end
    end
    
   
end

