classdef ClassificationHelper
    
    properties
    end
    
    methods
        function ch = ClassificationHelper()
        
        end
        
        function [training_indeces, test_indeces, val_indeces] = SplitData(~, dataset, training_percentage, test_percentage)
            ptrain = max(0, min(1, training_percentage));
            ptest = max(0, min(1, test_percentage));
            
            data_length = length(dataset);
            train_length = floor(ptrain * data_length);
            test_length = floor(ptest * data_length);
            
            indeces = 1:length(dataset);
            
            itrain = randsample(1:length(indeces), train_length);
            
            training_indeces = indeces(itrain);
            indeces(itrain) = [];
            
            
            
            itest = randsample(1:length(indeces), test_length);
            
            test_indeces = indeces(itest);
            indeces(itest) = [];
            
            
            val_indeces = indeces;
        end
        
        function targets = GenerateTargets(~, dataset, column)
            column = dataset(:, column);
            
            classes = sort( unique(column) );
            nclasses = length( classes );
            
            targets = zeros(length(column), nclasses);
            
            for ii = 1:nclasses
                c = classes(ii);
                targets(: , ii) = column == c;
            end
        end
        
        function [simout, icaout] = ProcessIntentionalModule(~, ia, input_index, im_index, dataset)
           

            data_length = length(dataset);
            module = ia.modules{im_index};
            
                
            [icaout, in_pca, sim, simout] = ProcessForward(ia, module, dataset);
                
           
        end
        
        function GScatter3(~, data, groups, fig_no, fig_title)
            
            
            ll = size(data, 2);
            
            ll = min(3, ll);
            
            for jj = ll+1:3
                data(:,jj) = zeros(length(data),1);
            end
            
            
            data = data(:, 1:3);
            
            
            
            figure(fig_no);
            
            groups_unique = unique(groups);
            hold on;
            for cc = 1:length(groups_unique)
                indeces = find(groups == groups_unique(cc));
                scatter3(data(indeces,1), data(indeces,2), data(indeces,3), '.');
            end
            grid on;
            hold off;
            title(fig_title);
        end
        
        function pc = DoPCA(~, data, var_perc)
           [pct.coeff, pct.score, pct.latent, pct.tsquared, pct.explained, pct.mu] = pca(data);

            perc = cumsum(pct.latent) / sum(pct.latent);

            perc(perc > var_perc) = 0;
            perc(perc > 0 ) = 1;
            perc(1) = 1;
            k = sum(perc);
            
            
            [pc.coeff, pc.score, pc.latent, pc.tsquared, pc.explained, pc.mu] = pca(data, 'NumComponents', k);
        end
        
        function [centroids, ids, meand] =  DoKmeans(~, points, k)
            [centroids, ids, meand, ~] = categorizeInput(points, k);
        end
        
        function [data, targets] = CreateDataset(~, class_proportions, data_length, sample_length)
           nclasses = length(class_proportions);
           
           targets = zeros(nclasses, data_length);
           
           rng = randsample(nclasses, data_length, true);
           
           
           ncomponents = 5;
           
           
           data = zeros(sample_length, data_length);
           
           coeff = floor(20 * rand(ncomponents, nclasses));
           
           variance = 5 * rand(ncomponents);
           
           for ii = 1:nclasses
              targets(ii, :) = rng == ii;
              
              class_samples = sum(rng == ii);
              
              
              c1 = repmat(4 .* sin(1:sample_length)', 1, class_samples) .* (variance(1) .* randn(sample_length, class_samples) + coeff(1,ii));
              c2 = repmat(.001 .* exp(1:sample_length)', 1, class_samples) .* (variance(2) .* randn(sample_length, class_samples) + coeff(2,ii));
              c3 = repmat(-5 .* (1:sample_length)', 1, class_samples) .* (variance(3) .* randn(sample_length, class_samples) + coeff(3,ii));
              c4 = repmat(1 .* cos(1:sample_length)', 1, class_samples) .* (variance(3) .* randn(sample_length, class_samples) + coeff(4,ii));
              c5 = repmat(1 .* log(1:sample_length)', 1, class_samples) .* (variance(3) .* randn(sample_length, class_samples) + coeff(4,ii));
              
              data(:, rng == ii) = c1 + c2 + c3 + c4 + c5;
           end
           
           
           
        end
    end
    
end

