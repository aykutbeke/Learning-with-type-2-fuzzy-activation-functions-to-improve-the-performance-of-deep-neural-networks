classdef SIT2FRU < nnet.layer.Layer
    
    properties (Learnable)
        a %FOU 
        b1 %Positive slope
        b2 %Negative slope
        Kf %Scale
    end
    properties 
%         Kf %Scale
    end
    methods
        function layer = SIT2FRU(num_channels,name)
            layer.Type = 'Single Input Interval Type2 Fuzzy Rectifying Unit';
            if nargin > 1
                layer.Name = name;
            end
            
            % Layer description.
            layer.Description = "Single Input Interval Type2 Fuzzy Unit with " + ...
                num_channels + " channels";
            
            % Initilization of learnable parameters
             layer = layer.initialize(num_channels, 0.5, 0.9, 0.9);
             layer.Kf = 1;
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
            
            %% Scale data into universe of discourse
            eps=10^-6;
%             Kf = layer.scale(Xi);
            X = Xi/layer.Kf;
            %% Saturations on FOU, if it is needed
            layer = layer.saturate(0.95, 0.05);
            %% Type-2 Fuzzy Gains
            Xp = max(0,X); Xn = min(0,X);
            Kp = 0.5*(layer.b1./(Xp - Xp.*layer.a + layer.a + eps)...
                - (layer.b1-layer.b1.*layer.a)./(Xp.*layer.a-1 + eps));
            Kn = 0.5*(layer.b2./(-Xn + Xn.*layer.a + layer.a + eps)...
                - (layer.b2-layer.b2.*layer.a)./(-Xn.*layer.a-1 + eps));
            
            %% Predict output
            Z = (Xp.*Kp + Xn.*Kn);
            Z = Z*layer.Kf; 
        end
        
%         function [dLdX, dLda, dLdb1, dLdb2] = backward(layer, x, ~, dLdZ, ~)
%             % Backward propagate the derivative of the loss function through
%             % the layer
%             %
%             % Inputs:
%             %         layer             - Layer to backward propagate through
%             %         x                 - Input data
%             %         Z                 - Output of layer forward function (not used)  
%             %         dLdZ              - Gradient propagated from the deeper layer
%             %         memory            - Memory value which can be used in (not used)
%             %                             backward propagation [unused]
%             % Output:
%             %         dLdX              - Derivative of the loss with
%             %                             respect to the input data
%             %         dLda              - Derivatives of the loss with
%             %                             respect to a(FOU size) 
%             %         dLdb1             - Derivatives of the loss with
%             %                             respect to bl(positive slope)
%             %         dLdb2             - Derivatives of the loss with
%             %                             respect to b2(negative slope)
%             
%             %% Get scaling factor
%             Kf = layer.scale(x);
%             sf = 1/Kf;
%             %% Saturations, if it is needed
%             layer = layer.saturate(0.95, 0.05);
%             %% Calculate derivatives    
%             %dLdX
%             eps=10^-6;
%             dfdX = layer.getdfdX(x, sf, eps);
%             dLdX = dLdZ.*dfdX;
%             %dLda
%             dfda = layer.getdfda(x, sf, eps);
%             dLda = dLdZ .* dfda;
%             %dLdb1 & dLdb2
%             dfdb1 = layer.getdfdb1(x, sf, eps);
%             dfdb2 = layer.getdfdb2(x, sf, eps);
%             dLdb1 = dLdZ .* dfdb1;
%             dLdb2 = dLdZ .* dfdb2;
%            
%             %% Sum of the derivatives over rows and colums
%             dLda = sum(sum(dLda,3),2);
%             dLdb1=sum(sum(dLdb1,3),2);
%             dLdb2=sum(sum(dLdb2,3),2);
% 
%         end
        
        function layer = initialize(layer,num_channels, alpha, slope1, slope2)
            layer.a =   (2.*rand(num_channels,1)-1)*0.1 + alpha;
            layer.b1 = (2.*rand(num_channels,1)-1)*0.1 + slope1; 
            layer.b2 = (2.*rand(num_channels,1)-1)*0.1 + slope2;
%             layer.a = ones(num_channels,1).*alpha;
%             layer.b1 = ones(num_channels,1).*slope1; 
%             layer.b2 = ones(num_channels,1).*slope2;
        end
        
        function kf = scale(layer, Xi)
            global Kf;
                eps = 0; 
                sf = max(max(max(max(max(max(0, Xi)),max(abs(min(0, Xi))))))) + eps;
                if Kf < sf
                    Kf = sf;
                end
                kf = gather(Kf);
        end
        
        function layer = saturate(layer, s1, s2)
                layer.a(layer.a > s1) = s1 ; 
                layer.a(layer.a < s2) = s2;
        end
        
        function dfdX = getdfdX(layer, x, sf, eps) 
            
            dfdx_p = layer.b1./(2*(layer.a + sf.*x - layer.a.*sf.*x)+eps) - (layer.b1 - layer.b1.*layer.a)./(2*(layer.a.*sf.*x - 1)+eps) ...
                - x.*((layer.b1.*(sf - layer.a.*sf))./(2*(layer.a + sf.*x - layer.a.*sf.*x).^2+eps) - ...
                (layer.a.*sf.*(layer.b1 - layer.b1.*layer.a))./(2*(layer.a.*sf.*x - 1).^2+eps));
            
            dfdx_n = x.*((layer.b2.*(sf - layer.a.*sf))./(2*(layer.a - sf.*x + layer.a.*sf.*x).^2+eps) - ...
                (layer.a.*sf.*(layer.b2 - layer.b2.*layer.a))./(2*(layer.a.*sf.*x + 1).^2+eps)) + (layer.b2 - layer.b2.*layer.a)./(2*(layer.a.*sf.*x + 1)+eps)...
                + layer.b2./(2*(layer.a - sf.*x + layer.a.*sf.*x)+eps);
            
            dfdx_p(x < 0) = 0; dfdx_n(x > 0) = 0; %setting 0 to not affect the summation 
            dfdX = (dfdx_p + dfdx_n);
        end
        
        function dfda = getdfda(layer, x, sf, eps)
            
            dfda_p = x.*(layer.b1./(2*(layer.a.*sf.*x - 1)+eps) + (layer.b1.*(sf.*x - 1))./(2*(layer.a + sf.*x - layer.a.*sf.*x).^2+eps)...
                + (sf.*x.*(layer.b1 - layer.b1.*layer.a))./(2*(layer.a.*sf.*x - 1).^2+eps));
            
            dfda_n = -x.*(layer.b2./(2*(layer.a.*sf.*x + 1)+eps) + (layer.b2.*(sf.*x + 1))./(2*(layer.a - sf.*x + layer.a.*sf.*x).^2+eps)...
                + (sf.*x.*(layer.b2 - layer.b2.*layer.a))./(2*(layer.a.*sf.*x + 1).^2+eps));

            
            dfda_p(x < 0) = 0; dfda_n(x > 0) = 0; %setting 0 to not affect the summation 
            dfda = dfda_p + dfda_n;
        end
        
        function dfdb1 = getdfdb1(layer, x, sf, eps)
            dfdb1 = x.*(1./(2*(layer.a + sf.*x - layer.a.*sf.*x)+eps) + (layer.a - 1)./(2*(layer.a.*sf.*x - 1)+eps));
            dfdb1(x<0)=0;
        end
        
        function dfdb2 = getdfdb2(layer, x, sf, eps)
            dfdb2 = x.*(1./(2*(layer.a - sf.*x + layer.a.*sf.*x)+eps) - (layer.a - 1)./(2*(layer.a.*sf.*x + 1)+eps));
            dfdb2(x>0)=0;
        end
        
    end
end