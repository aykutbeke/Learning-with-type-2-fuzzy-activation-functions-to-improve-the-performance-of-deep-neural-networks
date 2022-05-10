classdef SIT2FMLayerOpt < nnet.layer.Layer
    
    properties (Learnable)
        a
        b1
        b2
%         sfc
    end
    
%     properties (SetAccess = 'public')   
%         sfc =  single(1);
%     end
%     
    
    methods
       
        function layer = SIT2FMLayerOpt(num_channels,name)
            layer.Type = 'Single Input Interval Type2 Fuzzy Unit';
%             global scale;
            % Assign layer name if it is passed in.
            if nargin > 1
                layer.Name = name;
            end
            
            % Give the layer a meaningful description.
            layer.Description = "Single Input Interval Type2 Fuzzy Unit with " + ...
                num_channels + " channels";
            
%             layer.a = ones(1,1,num_channels)*0.5;
%             layer.b1 = 0.99*ones(1,1,num_channels); 
%             layer.b2 = 0.1*ones(1,1,num_channels);
            layer.a =   (2.*rand(num_channels,1)-1)*0.1 + 0.5;
            layer.b1 = (2.*rand(num_channels,1)-1)*0.1 + 0.92; 
            layer.b2 = (2.*rand(num_channels,1)-1)*0.1 + 0.15;
          
            
        end
        
        function Z = predict(layer,Xi)
            global scale;
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
%             X_p = max(0, Xi); X_n = min(0, Xi);
            sf = max(max(max(max(max(max(0, Xi)),max(abs(min(0, Xi)))))))+eps;
            if scale < sf
                scale = sf;
            end
            scale = gather(scale);
%             scale = 50;
%             if(scale > layer.sfc)
% %                 setSfc(double(scale))
%                 layer.sfc = scale;
% %                 layer=this.layer;
%             end
%             scale = layer.sfc;
            X = Xi/scale;
            X(X>1) = 1; X(X<-1) = -1;
            layer.a(layer.a>0.99) = 0.99 ; layer.a(layer.a<0.01) = 0.01;
%             layer.b1(layer.b1<0) = 0.001;  layer.b2(layer.b2<0) = 0.001;
%             layer.b1(layer.b1>1) = 1 ; layer.b2(layer.b2>1) = 1;
            
            %% Hard assigning 
%             layer.b1 = 1.5; 
%             layer.b2 = 0.0;
%             layer.a = 0.2;
         
            %% Assign Variables
%             B1  = layer.b1; B2 = layer.b2; alpha = layer.a;
            Xp = max(0,X); Xn = min(0,X);
            
            %% SIT2 Mapping
%             if max(max(max(max(isnan(X)))))||  max(max(max(max(isinf(X)))))
%                 a=1;
%             end
            Kp = 0.5*(layer.b1./(Xp - Xp.*layer.a + layer.a + eps)...
                - (layer.b1-layer.b1.*layer.a)./(Xp.*layer.a-1 + eps));
            Kn = 0.5*(layer.b2./(-Xn + Xn.*layer.a + layer.a + eps)...
                - (layer.b2-layer.b2.*layer.a)./(-Xn.*layer.a-1 + eps));
            
            Z = (Xp.*Kp + Xn.*Kn);
            
            Z = Z*scale;
            
%             if max(max(max(max(isnan(Z)))))||  max(max(max(max(isinf(Z)))))
%                 a=1;
%             end
%             
        end
        
        function [dLdX, dLda, dLdb1, dLdb2] = backward(layer, x, ~, dLdZ, ~)
            global scale;
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
%             X_p = max(0, Xi); X_n = min(0, Xi);
            sf = max(max(max(max(max(max(0, x)),max(abs(min(0, x)))))))+eps;
            if scale < sf
                scale = sf;
            end
            scale = gather(scale);
%             if(scale > layer.sfc)
%                 layer.sfc = scale;
%             end
%             scale = layer.sfc;
%             scale = 50;
%             X = Xi/scale;
%             X = Xi;
%             x = Xi;
            sf = 1/scale;
%             X(X>1) = 1; X(X<-1) = -1;
            
           layer.a(layer.a>0.99) = 0.99 ; layer.a(layer.a<0.01) = 0.01;
%            layer.b1(layer.b1<0) = 0.001;  layer.b2(layer.b2<0) = 0.001;
%            layer.b1(layer.b1>1) = 1 ; layer.b2(layer.b2>1) = 1;
            %% Hard assigning
%             layer.b1 = 1.5;
%             layer.b2 = 0.0; 
%             layer.a = 0.2;

            %% Assign variables
%             layerB1  = layer.b1; layerb2 = layer.b2; layera = layer.a; layera = layer.a;
%             sigma_p =  max(0, X); sigma_n = min(0, X);
           
            %% dLdX
%            dfdx_p = B1./(2*(alpha1 + sigma_p - alpha1.*sigma_p) + eps) ...
%                 + sigma_p.*((B1.*(alpha1 - 1))./(2*(alpha1 + sigma_p - alpha1.*sigma_p).^2 + eps)...
%                 + (alpha1.*(B1 - B1.*alpha1))./(2*(alpha1.*sigma_p - 1).^2 + eps)) - (B1 - B1.*alpha1)./(2*(alpha1.*sigma_p - 1) + eps);
%             
%             dfdx_n = B2./(2*(alpha2 - sigma_n + alpha2.*sigma_n) + eps)...
%                 - sigma_n.*((alpha2.*(B2 - B2.*alpha2))./(2*(alpha2.*sigma_n + 1).^2 + eps)...
%                 + (B2.*(alpha2 - 1))./(2*(alpha2 - sigma_n + alpha2.*sigma_n).^2 + eps)) + (B2 - B2.*alpha2)./(2*(alpha2.*sigma_n + 1) + eps);
%             
            
            dfdx_p = layer.b1./(2*(layer.a + sf.*x - layer.a.*sf.*x)+eps) - (layer.b1 - layer.b1.*layer.a)./(2*(layer.a.*sf.*x - 1)+eps) ...
                - x.*((layer.b1.*(sf - layer.a.*sf))./(2*(layer.a + sf.*x - layer.a.*sf.*x).^2+eps) - ...
                (layer.a.*sf.*(layer.b1 - layer.b1.*layer.a))./(2*(layer.a.*sf.*x - 1).^2+eps));
            
            dfdx_n = x.*((layer.b2.*(sf - layer.a.*sf))./(2*(layer.a - sf.*x + layer.a.*sf.*x).^2+eps) - ...
                (layer.a.*sf.*(layer.b2 - layer.b2.*layer.a))./(2*(layer.a.*sf.*x + 1).^2+eps)) + (layer.b2 - layer.b2.*layer.a)./(2*(layer.a.*sf.*x + 1)+eps)...
                + layer.b2./(2*(layer.a - sf.*x + layer.a.*sf.*x)+eps);

            
            dfdx_p(x<0) = 0; dfdx_n(x>0) = 0;
            dfdX = (dfdx_p + dfdx_n);
            clear dfdx_p dfdx_n
            dLdX = dLdZ.*dfdX;
            
            %% dLdalpha
%             dfda_p = sigma_p.*(B1./(2*(alpha1.*sigma_p - 1) + eps)...
%                 + (B1.*(sigma_p - 1))./(2*(alpha1 + sigma_p - alpha1.*sigma_p).^2 + eps) ...
%                 + (sigma_p.*(B1 - B1.*alpha1))./(2*(alpha1.*sigma_p - 1).^2 + eps));
            
%             dfda_n = -sigma_n.*(B2./(2*(alpha2.*sigma_n + 1) + eps)...
%                 + (sigma_n.*(B2 - B2.*alpha2))./(2*(alpha2.*sigma_n + 1).^2 + eps)...
%                 + (B2.*(sigma_n + 1))./(2*(alpha2 - sigma_n + alpha2.*sigma_n).^2 + eps));
            
            dfda_p = x.*(layer.b1./(2*(layer.a.*sf.*x - 1)+eps) + (layer.b1.*(sf.*x - 1))./(2*(layer.a + sf.*x - layer.a.*sf.*x).^2+eps)...
                + (sf.*x.*(layer.b1 - layer.b1.*layer.a))./(2*(layer.a.*sf.*x - 1).^2+eps));
            
            dfda_n = -x.*(layer.b2./(2*(layer.a.*sf.*x + 1)+eps) + (layer.b2.*(sf.*x + 1))./(2*(layer.a - sf.*x + layer.a.*sf.*x).^2+eps)...
                + (sf.*x.*(layer.b2 - layer.b2.*layer.a))./(2*(layer.a.*sf.*x + 1).^2+eps));

            
            dfda_p(x<0)=0; dfda_n(x>0)=0;
            dfda = dfda_p + dfda_n;
            clear dfda_p dfda_n
            dLda = dLdZ .* dfda;
            
            %% dLdb
            dfdb1 = x.*(1./(2*(layer.a + sf.*x - layer.a.*sf.*x)+eps) + (layer.a - 1)./(2*(layer.a.*sf.*x - 1)+eps));
            dfdb2 = x.*(1./(2*(layer.a - sf.*x + layer.a.*sf.*x)+eps) - (layer.a - 1)./(2*(layer.a.*sf.*x + 1)+eps));

%             dfdb1 = sigma_p.*((alpha1 - 1)./(2*(alpha1.*sigma_p - 1) + eps) + 1./(2*(alpha1 + sigma_p - alpha1.*sigma_p) + eps));
%             dfdb2 = -sigma_n.*((alpha2 - 1)./(2*(alpha2.*sigma_n + 1) + eps) - 1./(2*(alpha2 - sigma_n + alpha2.*sigma_n) + eps));
            
            dfdb1(x<0)=0; dfdb2(x>0)=0;
           
            dLdb1 = dLdZ .* dfdb1;
            dLdb2 = dLdZ .* dfdb2;
            clear dfdb1 dfdb2
            [sz] = size(dLda);
%             % Sum over the image rows and columns.
            dLda = sum(sum(dLda,3),2);
%             % Sum over all the observations in the mini-batch.
%             dLda = sum(dLda,length(sz));
% %             
%             [sz] = size(dLdb1);
% %             % Sum over the image rows and columns.
%             dLdb1 = sum(sum(dLdb1,1),2);
% %             % Sum over all the observations in the mini-batch.
%             dLdb1 = sum(dLdb1,length(sz));
             dLdb1=sum(sum(dLdb1,3),2);
             dLdb2=sum(sum(dLdb2,3),2);
%             [sz] = size(dLdb2);
% %             % Sum over the image rows and columns.
%             dLdb2 = sum(sum(dLdb2,1),2);
% %             % Sum over all the observations in the mini-batch.
%             dLdb2 = sum(dLdb2,length(sz));
%             
%             dLdsf = dLdZ.*0;
%             dLdsf = sum(sum(dLdsf,1),2);
%             % Sum over all the observations in the mini-batch.
%             dLdsf = sum(sum(dLdsf,4));
            
%             if max(max(max(max(isnan(dLda))))) ||  max(max(max(max(isnan(dLdX)))))
%                 a=1;
%             end
        end
        
%         function layer = SIT2FMLayerOpt(val)
%             val.sfc = scale; 
%             this.layer = val;
%         end
    end
end