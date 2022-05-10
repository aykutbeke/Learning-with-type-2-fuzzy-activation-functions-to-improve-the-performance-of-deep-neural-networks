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
            eps =0;
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
            if  max(max(max(max(isnan(X)))))||  max(max(max(max(isinf(X)))))
             a = 1;
            end
            
            X_p = max(0, X);
            X_n = min(0, X);
%             scale = max(max(X_p),max(abs(X_n)));
            scale = 255;
%             Z = X_p./(scale+eps) + X_n./(scale+eps);   
            Z = X./(scale+eps);
%             Z(isnan(Z)) = 0;  Z(isinf(Z)) = 1;
            if  max(max(max(max(isnan(Z)))))||  max(max(max(max(isinf(Z)))))
             a = 1;
            end
        end
        
        function [dLdX] = backward(layer, X, Z, dLdZ, ~)
           dLdX=dLdZ;
        end
    end
end