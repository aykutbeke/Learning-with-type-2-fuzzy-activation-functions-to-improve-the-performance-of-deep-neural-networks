classdef singleInputT2Layer < nnet.layer.Layer

    properties (Learnable)
        alpha
    end
    
    methods
        function layer = singleInputT2Layer(num_channels,name)
            layer.Type = 'Single Input Type2 Unit';
            
            % Assign layer name if it is passed in.
            if nargin > 1
                layer.Name = name;
            end
            
            % Give the layer a meaningful description.
            layer.Description = "Single Input Type2 Unit with " + ...
                num_channels + " channels";
            
            % Initialize the learnable alpha parameter.
            layer.alpha = ones(1,1,num_channels)*0.5;
        end

        function Z = predict(layer,X_i)
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
            in = layer.alpha < 0.05;
            ip = layer.alpha > 0.95;
          
            layer.alpha(in) = 0.05;
            layer.alpha(ip) = 0.95;
          
            
            layer.alpha(ip) = 1;
            X = abs(X_i);
            K = 1/2*(((layer.alpha + X) - (layer.alpha .* X + eps)).^-1 + ((layer.alpha-1)./(layer.alpha .* X-1+eps)));
            Z = X.*K; 
            Z(X_i>0) = X_i(X_i>0);
            Z(X_i<0) = -Z(X_i<0);
        end
        
        function [dLdX, dLdAlpha] = backward(layer, X_i, Z_i, dLdZ, ~)
            eps = 10^-6; 
            % Backward propagate the derivative of the loss function through 
            % the layer
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X                 - Input data
            %         Z                 - Output of layer forward function            
            %         dLdZ              - Gradient propagated from the deeper layer
            %         memory            - Memory value which can be used in
            %                             backward propagation [unused]
            % Output:
            %         dLdX              - Derivative of the loss with
            %                             respect to the input data
            %         dLdAlpha          - Derivatives of the loss with
            %                             respect to alpha
            X = abs(X_i);
            dZdX = (layer.alpha - 1)./(layer.alpha.*X - 1 + eps) + ...
                X.*((layer.alpha - 1)./(layer.alpha + X - layer.alpha.*X + eps).^2 - ...
                (layer.alpha.*(layer.alpha - 1))./(layer.alpha.*X - 1 + eps).^2) + ...
                1./(layer.alpha + X - layer.alpha.*X + eps);
            dZdX(X_i<0) = -dZdX(X_i<0);
            dZdX = dZdX/2;
            dLdX = dLdZ.*dZdX;
            dLdX(X_i>0) = dLdZ(X_i>0);
            
            dZdAlpha = X.*(1./(layer.alpha.*X - 1 + eps) + (X - 1)./(layer.alpha + X - layer.alpha.*X + eps).^2 ...
                - (X.*(layer.alpha - 1))./(layer.alpha.*X - 1 + eps).^2);
            dZdAlpha(X_i<0) = -dZdAlpha(X_i<0);
            dZdAlpha = dZdAlpha/2;
            dLdAlpha =  dZdAlpha.*dLdZ;
            % Sum over the image rows and columns.
            dLdAlpha = sum(sum(dLdAlpha,1),2);
            % Sum over all the observations in the mini-batch.
            dLdAlpha = sum(dLdAlpha,4);
        end
    end
end