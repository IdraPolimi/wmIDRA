function [ mean_values, variance ] = naiveClassification( weights_set )
    mean_values = mean(weights_set);
    variance = var(weights_set);
end

