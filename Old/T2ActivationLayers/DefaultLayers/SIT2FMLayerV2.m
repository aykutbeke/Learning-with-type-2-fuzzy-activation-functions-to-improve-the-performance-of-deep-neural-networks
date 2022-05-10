classdef SIT2FMLayerV2 < nnet.layer.Layer
    
    properties (Learnable)
        a1
        a2
        b1
        b2
    end
    
%      properties
%          b1
%          b2
%     end
     
    methods
        function layer = SIT2FMLayerV2(num_channels,name)
            layer.Type = 'Single Input Interval Type2 Fuzzy Unit';
            
            % Assign layer name if it is passed in.
            if nargin > 1
                layer.Name = name;
            end
            
            % Give the layer a meaningful description.
            layer.Description = "Single Input Interval Type2 Fuzzy Unit with " + ...
                num_channels + " channels";
            
%             layer.b1 = 5*ones(1,1,num_channels); 
            layer.a1 = ones(1,1,num_channels)*0.5;
            layer.a2 = ones(1,1,num_channels)*0.5;
            layer.b1 = 0.9*ones(1,1,num_channels); 
            layer.b2 = 0.1*ones(1,1,num_channels); 

        end
        
        function Z = predict(layer,Xi)
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
            
            %% Saturations and normalizations, if it is needed
            eps = 0; 
            X_p = max(0, Xi); X_n = min(0, Xi);
            scale = max(max(max(max(max(X_p),max(abs(X_n))))));
%             scale = 10; 
            X = Xi/scale;
            
            layer.a1(layer.a1>0.95) = 0.95 ; layer.a1(layer.a1<0.05) = 0.05;
            layer.a2(layer.a2>0.95) = 0.95 ; layer.a2(layer.a2<0.05) = 0.05;
%             layer.b1(layer.b1<=0) = 0.001;  layer.b2(layer.b2<0) = 0.001;
%             layer.b1(layer.b1>1) = 0.99 ; layer.b2(layer.b2>1) = 0.99;
            
            %% Hard assigning 
%             layer.b1 = 1 ; 
%             layer.b2 = 0.01;
%             layer.a = 0.5;
%             layer.b1 = 1*ones(1,1,num_channels); 
%             layer.b2 = 0.1*ones(1,1,num_channels); 
            
            %% Assign Variables
            B1  = layer.b1; B2 = layer.b2; alpha1 = layer.a1; alpha2 = layer.a2;
            Xp = max(0,X); Xn = min(0,X);
            
            %% SIT2 Mapping
            if max(max(max(max(isnan(X)))))||  max(max(max(max(isinf(X)))))
                a=1;
            end
            Kp = 0.5*(B1./(Xp - Xp.*alpha1 + alpha1 + eps)...
                - (B1-B1.*alpha1)./(Xp.*alpha1-1 + eps));
            Kn = 0.5*(B2./(-Xn + Xn.*alpha2 + alpha2 + eps)...
                - (B2-B2.*alpha2)./(-Xn.*alpha2-1 + eps));
            
            Z = (Xp.*Kp + Xn.*Kn);
            
            Z = Z*scale;
            
            if max(max(max(max(isnan(Z)))))||  max(max(max(max(isinf(Z)))))
                a=1;
            end
            
        end
        
        function [dLdX, dLda1, dLda2, dLdb1, dLdb2] = backward(layer, Xi, Z, dLdZ, ~)
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
            
            %% Saturations and normalizations, if it is needed
            eps = 0;
            X_p = max(0, Xi); X_n = min(0, Xi);
            scale = max(max(max(max(max(X_p),max(abs(X_n))))));
%             scale = 10; 
            X = Xi;
            x = Xi;
            sf = 1/scale;
            
           layer.a1(layer.a1>0.95) = 0.95 ; layer.a1(layer.a1<0.05) = 0.05;
           layer.a2(layer.a2>0.95) = 0.95 ; layer.a2(layer.a2<0.05) = 0.05;
%            layer.b1(layer.b1<=0) = 0.001;  layer.b2(layer.b2<0) = 0.001;
%            layer.b1(layer.b1>1) = 0.99 ; layer.b2(layer.b2>1) = 0.99;
            %% Hard assigning
%             layer.b1 = 1 ;
%             layer.b2 = 0.01; 

           
            %% Assign variables
            B1  = layer.b1; B2 = layer.b2; alpha1 = layer.a1; alpha2 = layer.a2;
%             sigma_p =  max(0, X); sigma_n = min(0, X);
           
            %% dLdX
%             dfdx_p = B1./(2*(alpha1 + sigma_p - alpha1.*sigma_p) + eps) ...
%                 + sigma_p.*((B1.*(alpha1 - 1))./(2*(alpha1 + sigma_p - alpha1.*sigma_p).^2 + eps)...
%                 + (alpha1.*(B1 - B1.*alpha1))./(2*(alpha1.*sigma_p - 1).^2 + eps)) - (B1 - B1.*alpha1)./(2*(alpha1.*sigma_p - 1) + eps);
%             
%             dfdx_n = B2./(2*(alpha2 - sigma_n + alpha2.*sigma_n) + eps)...
%                 - sigma_n.*((alpha2.*(B2 - B2.*alpha2))./(2*(alpha2.*sigma_n + 1).^2 + eps)...
%                 + (B2.*(alpha2 - 1))./(2*(alpha2 - sigma_n + alpha2.*sigma_n).^2 + eps)) + (B2 - B2.*alpha2)./(2*(alpha2.*sigma_n + 1) + eps);
%             
             dfdx_p = B1./(2*(alpha1 + sf.*x - alpha1.*sf.*x)+eps) - (B1 - B1.*alpha1)./(2*(alpha1.*sf.*x - 1)+eps) ...
                - x.*((B1.*(sf - alpha1.*sf))./(2*(alpha1 + sf.*x - alpha1.*sf.*x).^2+eps) - ...
                (alpha1.*sf.*(B1 - B1.*alpha1))./(2*(alpha1.*sf.*x - 1).^2+eps));
            
            dfdx_n = x.*((B2.*(sf - alpha2.*sf))./(2*(alpha2 - sf.*x + alpha2.*sf.*x).^2+eps) - ...
                (alpha2.*sf.*(B2 - B2.*alpha2))./(2*(alpha2.*sf.*x + 1).^2+eps)) + (B2 - B2.*alpha2)./(2*(alpha2.*sf.*x + 1)+eps)...
                + B2./(2*(alpha2 - sf.*x + alpha2.*sf.*x)+eps);
            
            dfdx_p(X<0) = 0; dfdx_n(X>0) = 0;
            dfdX = (dfdx_p + dfdx_n);
            
            dLdX = dLdZ.*dfdX;
            
            %% dLdalpha
            dfda_p = x.*(B1./(2*(alpha1.*sf.*x - 1)+eps) + (B1.*(sf.*x - 1))./(2*(alpha1 + sf.*x - alpha1.*sf.*x).^2+eps)...
                + (sf.*x.*(B1 - B1.*alpha1))./(2*(alpha1.*sf.*x - 1).^2+eps));
            
            dfda_n = -x.*(B2./(2*(alpha2.*sf.*x + 1)+eps) + (B2.*(sf.*x + 1))./(2*(alpha2 - sf.*x + alpha2.*sf.*x).^2+eps)...
                + (sf.*x.*(B2 - B2.*alpha2))./(2*(alpha2.*sf.*x + 1).^2+eps));

            
            dfda_p(X<0)=0; dfda_n(X>0)=0;
%             dfda = dfda_p + dfda_n;
            
            dLda1 = dLdZ .* dfda_p;
            dLda2 = dLdZ .* dfda_n;
            %% dLdb
            dfdb1 = x.*(1./(2*(alpha1 + sf.*x - alpha1.*sf.*x)+eps) + (alpha1 - 1)./(2*(alpha1.*sf.*x - 1)+eps));
            dfdb2 = x.*(1./(2*(alpha2 - sf.*x + alpha2.*sf.*x)+eps) - (alpha2 - 1)./(2*(alpha2.*sf.*x + 1)+eps));

            dfdb1(X<0)=0; dfdb2(X>0)=0;
           
            dLdb1 = dLdZ .* dfdb1;
            dLdb2 = dLdZ .* dfdb2;
            
%             scale=1;
            % Sum over the image rows and columns.
            dLda1 = sum(sum(dLda1,1),2);
            % Sum over all the observations in the mini-batch.
            dLda1 = sum(dLda1,4);
            
            % Sum over the image rows and columns.
            dLda2 = sum(sum(dLda2,1),2);
            % Sum over all the observations in the mini-batch.
            dLda2 = sum(dLda2,4);
            
%             Sum over the image rows and columns.
            dLdb1 = sum(sum(dLdb1,1),2);
            % Sum over all the observations in the mini-batch.
            dLdb1 = sum(dLdb1,4);
%             dLdb1(layer.b1 == 1) = 0;
            
            % Sum over the image rows and columns.
            dLdb2 = sum(sum(dLdb2,1),2);
            % Sum over all the observations in the mini-batch.
            dLdb2 = sum(dLdb2,4);
            
            if max(max(max(max(isnan(dLda1))))) ||  max(max(max(max(isnan(dLdX)))))
                a=1;
            end
        end
    end
end