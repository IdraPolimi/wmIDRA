classdef FilterNode < handle
    %INPUTNODE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        index
        ia
        InputSize
        filterFunction
    end
    
    methods
        function node = FilterNode(intentionalArchitecture, index, InputSize, filterFunction)
            node.ia = intentionalArchitecture;
            node.index = index;
            node.InputSize = InputSize;
            node.filterFunction = filterFunction;
        end
        
        function SetInput(node, values)
            
            values = node.filterFunction(values);
            
            if length(values) < node.InputSize
                values = [values, zeros(1, node.InputSize - length(values))];
            end
            
            values = values(:, 1 : node.InputSize);
            
            node.ia.SetModuleInput(node.index, values);
        end
        
        function size = get.InputSize(node)
            size = node.InputSize;
        end
    end
    
end

