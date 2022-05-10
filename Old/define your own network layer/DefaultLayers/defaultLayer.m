classdef defaultLayer < nnet.layer.Layer
    
    properties 
    num_reluNeuron
    num_eluNeuron
    end
    
    properties (Learnable)
        % Layer learnable parameters
        
        % Scaling coefficient
        Alpha1
        Alpha2
    end
    
    methods
        function layer = defaultLayer(numReluNeuron, numEluNeuron, name)
            % Create an examplePreluLayer with numChannels channels
            
            % Set layer name
            if nargin == 3
                layer.Name = name;
            end
            
            % Set layer description
            layer.Description = ...
                ['exampledefaultLayer with ', num2str(numReluNeuron+numEluNeuron), ' channels'];
            
            % Initialize scaling coefficient
            layer.num_reluNeuron = numReluNeuron;
            layer.num_eluNeuron = numEluNeuron;
            layer.Alpha1 = rand([1 1 numReluNeuron]);
            layer.Alpha2 = rand([1 1 numEluNeuron]);
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer and output the result
            
            %% Relu Output
            x1 = X(:,:,1:layer.num_reluNeuron);
            x2 = X(:,:,layer.num_eluNeuron+1:end);
            z1 = max(0, x1) + layer.Alpha1 .* min(0, x1);
            z2 = (x2 .* (x2 > 0)) + ...
                (layer.Alpha2.*(exp(min(x2,0)) - 1) .* (x2 <= 0));
            
            Z = cat(3,z1,z2);
        end
        
        function [dLdX, dLdAlpha1, dLdAlpha2] = backward(layer, X, Z, dLdZ, memory) 
            % Backward propagate the derivative of the loss function through 
            % the layer 
            x1 = X(:,:,1:layer.num_reluNeuron);
            x2 = X(:,:,layer.num_eluNeuron+1:end);
            dLdz1 = dLdZ(:,:,1:layer.num_reluNeuron);
            dLdz2 = dLdZ(:,:,layer.num_eluNeuron+1:end);
            z1 = Z(:,:,1:layer.num_reluNeuron);
            z2 = Z(:,:,layer.num_eluNeuron+1:end);
            
            
            dLdx1 = layer.Alpha1 .* dLdz1;
            dLdx1(x1>0) = dLdz1(x1>0);
            dLdAlpha1 = min(0,x1) .* dLdz1;
            dLdAlpha1 = sum(sum(dLdAlpha1,1),2);
            % Sum over all observations in mini-batch
            dLdAlpha1 = sum(dLdAlpha1,4);
            
            
            dLdx2 = dLdz2 .* ((x2 > 0) + ...
                ((layer.Alpha2 + z2) .* (x2 <= 0)));            
            
            dLdAlpha2 = exp(min(x2,0) - 1) .* dLdz2;
            % Sum over the image rows and columns.
            dLdAlpha2 = sum(sum(dLdAlpha2,1),2);
            % Sum over all the observations in the mini-batch.
            dLdAlpha2 = sum(dLdAlpha2,4);
            
            dLdX = cat(3,dLdx1,dLdx2);
        end
    end
end