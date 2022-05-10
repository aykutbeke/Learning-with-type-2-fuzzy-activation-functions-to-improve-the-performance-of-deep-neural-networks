classdef SIT2FMLayer < nnet.layer.Layer

    properties (Learnable)
        a
        b1
        b2
    end
    
    methods
        function layer = SIT2FMLayer(num_channels,name)
            layer.Type = 'Single Input Interval Type2 Fuzzy Unit';
            
            % Assign layer name if it is passed in.
            if nargin > 1
                layer.Name = name;
            end
            
            % Give the layer a meaningful description.
            layer.Description = "Single Input Interval Type2 Fuzzy Unit with " + ...
                num_channels + " channels";
            
            layer.b1 = rand(1,1,num_channels); layer.a = rand(1,1,num_channels);
            layer.b2 = rand(1,1,num_channels);
        end

        function Z = predict(layer,X)
            eps = 10^-10; 
%             layer.b1 = ones(1,1,20);layer.b2 = ones(1,1,20)*0.1;
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
            layer.a(layer.a>0.95) = 0.95 ; layer.a(layer.a<0.05) = 0.05;
%             layer.a2(layer.a2>0.95) = 0.95 ; layer.a2(layer.a2<0.05) = 0.05;
              layer.b1(layer.b1<0) = 0.01;  layer.b2(layer.b2<0) = 0.01;
%             layer.b2(layer.b2>0.95) = 0.95 ; layer.b2(layer.b2<0.05) = 0.05;

            Xp = max(0,X);
            Xn = min(0,X);
            X = abs(X);
            Kp = 0.5*(layer.b1./(X - X.*layer.a + layer.a + eps)...
                    - (layer.b1-layer.b1.*layer.a)./(X.*layer.a-1 + eps));
            
            Kn = 0.5*(layer.b2./(X - X.*layer.a + layer.a + eps)...
                    - (layer.b2-layer.b2.*layer.a)./(X.*layer.a-1 + eps));
            
            Z = Xp.*Kp + Xn .*Kn;    
%             if sum(sum(sum(max(isnan(Z)))))>0
%                 a=1
%             end
            
            Z(isnan(Z)) = 0;
        end
        
        function [dLdX, dLda, dLdb1, dLdb2] = backward(layer, X1, Z, dLdZ, ~)
            eps = 10^-10; 
             layer.a(layer.a>0.95) = 0.95 ; layer.a(layer.a<0.05) = 0.05;
             layer.b1(layer.b1<0) = 0.01;  layer.b2(layer.b2<0) = 0.01;

%             layer.b1 = ones(1,1,20);layer.b2 = ones(1,1,20)*0.1;
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
         
            
            
            X = abs(X1);
            
            %% dLdX
            dfdx_p = layer.b1./(2*(layer.a - X + layer.a.*X) + eps)... 
                -X.*((layer.a.*(layer.b1 - layer.b1.*layer.a))./(2*(layer.a.*X + 1).^2 + eps) + (layer.b1.*(layer.a - 1))./(2*(layer.a - X + layer.a.*X).^2 + eps))...
                + (layer.b1 - layer.b1.*layer.a)./(2*(layer.a.*X + 1) + eps);
            
            dfdx_n = -(layer.b2./(2*(layer.a - X + layer.a.*X) + eps)... 
                -X.*((layer.a.*(layer.b2 - layer.b2.*layer.a))./(2*(layer.a.*X + 1).^2 + eps) + (layer.b2.*(layer.a - 1))./(2*(layer.a - X + layer.a.*X).^2 + eps))...
                + (layer.b2 - layer.b2.*layer.a)./(2*(layer.a.*X + 1) + eps));
            
            dfdx_p(X1<0) = 0;
            
            dfdx_n(X1>0) = 0;
            
            dfdX = dfdx_p + dfdx_n;
            
            dLdX = dLdZ.*dfdX;
           
           %% dLdalpha
            dfda_p = -X.*(layer.b1./(2*(layer.a.*X + 1) + eps)...
               + (X.*(layer.b1 - layer.b1.*layer.a))./(2*(layer.a.*X + 1).^2 + eps)...
               + (layer.b1.*(X + 1))./(2*(layer.a - X + layer.a.*X).^2 + eps));

            dfda_n = X.*(layer.b2./(2*(layer.a.*X + 1) + eps)...
               + (X.*(layer.b2 - layer.b2.*layer.a))./(2*(layer.a.*X + 1).^2 + eps)...
               + (layer.b2.*(X + 1))./(2*(layer.a - X + layer.a.*X).^2 + eps));
           
            dfda_p(X1<0)=0;
            
            dfda_n(X1>0)=0;
            
            dfda =dfda_p + dfda_n;
           
            %% dLdb
            dfdb1 = -X.*((layer.a - 1)./(2*(layer.a.*X + 1) + eps)...
               - 1./(2*(layer.a - X + layer.a.*X) + eps));
           
            dfdb2 = X.*((layer.a - 1)./(2*(layer.a.*X + 1) + eps)...
               - 1./(2*(layer.a - X + layer.a.*X) + eps));
          
           dfdb1(X1<0)=0;
           dfdb2(X1>0)=0;
           
          
           dLda = dLdZ .* dfda;
           dLdb1 = dLdZ .* dfdb1;
           dLdb2 = dLdZ .* dfdb2;
            

            % Sum over the image rows and columns.
            dLda = sum(sum(dLda,1),2);
            % Sum over all the observations in the mini-batch.
            dLda = sum(dLda,4);

             % Sum over the image rows and columns.
            dLdb1 = sum(sum(dLdb1,1),2);
            % Sum over all the observations in the mini-batch.
            dLdb1 = sum(dLdb1,4);

             % Sum over the image rows and columns.
            dLdb2 = sum(sum(dLdb2,1),2);
            % Sum over all the observations in the mini-batch.
            dLdb2 = sum(dLdb2,4);

            dLda(isnan(dLda)) = 0;
            dLdb1(isnan(dLdb1)) = 0;
            dLdb2(isnan(dLdb2)) = 0;
            dLdX(isnan(dLdX)) = 0;
 
        end
    end
end