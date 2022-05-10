classdef fisLayer < nnet.layer.Layer

    properties (Learnable)
        %% A1 
        a1;c1;
        %% A2
        a2;c2;
        %% B1 
        a3;c3;
        % B2 
        a4;c4;
        %% Consequent parameters (f1&f2) 
        r1; r2;
    end
    
    methods
        function layer = fisLayer(num_channels, name)
            layer.Type = 'fis Unit';
            
            % Assign layer name if it is passed in.
            if nargin > 1
                layer.Name = name;
            end
            
            % Give the layer a meaningful description.
            layer.Description = "Fis unit with " + ...
                num_channels + " channels";
            
            % Initialize the learnable alpha parameter.
            %% A1
            layer.a1 = rand(1,1,num_channels/2);layer.c1 = rand(1,1,num_channels/2);
            %% A2
            layer.a2 = rand(1,1,num_channels/2);layer.c2 = rand(1,1,num_channels/2);
            %% B1
            layer.a3 = rand(1,1,num_channels/2);layer.c3 = rand(1,1,num_channels/2);
            %% B2
            layer.a4 = rand(1,1,num_channels/2);layer.c4 = rand(1,1,num_channels/2);
            %% Consequent r1, r2
            layer.r1 =  rand(1,1,num_channels/2); layer.r2 =  rand(1,1,num_channels/2); 
        end

        function Z = predict(layer, X, Y)
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
            
            %% ForwardPass two-input Fis 
            [~, ~, s3] = size(X);
%             x = X(:,:,1:s3/2);
%             y = X(:,:,s3/2+1:end);
            x = X;
            y = Y;
            % two-input Fis Layer1
            mu_A1 = gaussmf(x, [layer.a1 layer.c1]); mu_A2 = gaussmf(x, [layer.a2 layer.c2]);
            mu_B1 = gaussmf(y, [layer.a3 layer.c3]); mu_B2 = gaussmf(y, [layer.a4 layer.c4]);
            % two-input Fis Layer2
            w1 = mu_A1.*mu_B1; w2 = mu_A2.*mu_B2;
            % two-input Fis Layer3
            w1_n = w1./(w1+w2); w2_n = w2./(w1+w2);
            % two-input Fis Layer4 && Layer5
            f1 = layer.r1; f2 = layer.r2;
            f = w1_n .* f1 + w2_n .* f2;
            %% output
            Z = f;
        end
        
        function [dLdX, dLdY ,dLda1,dLdc1, dLda2, dLdc2, dLda3, dLdc3, dLda4, dLdc4, dLdr1, dLdr2 ] = backward(layer, X, Z, dLdZ, ~)
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
            
           dLdX, dLdY, dLda1,dLdc1, dLda2, dLdc2, dLda3, dLdc3, dLda4, dLdc4, dLdr1, dLdr2
        end
    end
end