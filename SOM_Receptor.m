classdef SOM_Receptor < handle
    
   properties
      som 
      index
   end
   
   methods
       function obj = SOM_Receptor(som, index)
           obj.som = som;
           obj.index = index;
       end
   end
   
   methods
       function SetActivation(obj, val)
           obj.som.SetActivations(obj.index, val);
       end
       
       function a = GetActivation(obj)
           a = obj.som.GetActivations(obj.index);
       end
   end
end