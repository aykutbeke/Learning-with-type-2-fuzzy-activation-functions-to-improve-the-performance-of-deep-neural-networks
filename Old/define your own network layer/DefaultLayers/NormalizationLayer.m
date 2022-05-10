classdef NormalizationLayer < nnet.layer.Layer

  
    methods
        function layer = NormalizationLayer()
            layer.Type = 'Normalization';
            
            % Assign layer name if it is passed in.
            if nargin > 1
                layer.Name = name;
            end
            
            % Give the layer a meaningful description.
            layer.Description = "Normalization "; 
            
        end

        function Z = predict(layer,X)
            eps = 10^-6;
            % Forward input data through the layer at prediction time and
            % output the result
            %
            % Inputs:
            %         layer    -    Layer to forward propagate through
            %         X        -    Input data
            % Output:
            %         Z        -    Output of layer forward function
            
            % Expressing the computation in vectorized form allows it to
            % execute directly on the GPU.
            X_p = max(0, X);
            X_n = min(0, X);
            scale = max(max(X_p),max(abs(X_n)));
            
            Z = X_p./(scale) + X_n./(scale);
           
        end
        
        function [dLdX] = backward(layer, X, Z, dLdZ, ~)
           dLdX=dLdZ;
        end
    end
end