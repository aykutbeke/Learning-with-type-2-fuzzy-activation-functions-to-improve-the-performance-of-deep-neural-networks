function [dfdx, dfda] = dfdX(X1,a1,b1,b2)
            eps = 0;
            layer.a = a1;
            layer.b1 =b1;
            layer.b2 = b2;

          
            X = abs(X1);
            
            dfdx_p = layer.b1./(2*(layer.a - X + layer.a.*X))... 
                -X.*((layer.a.*(layer.b1 - layer.b1.*layer.a))./(2*(layer.a.*X + 1).^2) + (layer.b1.*(layer.a - 1))./(2*(layer.a - X + layer.a.*X).^2))...
                + (layer.b1 - layer.b1.*layer.a)./(2*(layer.a.*X + 1));
            
            dfdx_n = -(layer.b2./(2*(layer.a - X + layer.a.*X))... 
                -X.*((layer.a.*(layer.b2 - layer.b2.*layer.a))./(2*(layer.a.*X + 1).^2) + (layer.b2.*(layer.a - 1))./(2*(layer.a - X + layer.a.*X).^2))...
                + (layer.b2 - layer.b2.*layer.a)./(2*(layer.a.*X + 1)));
            
            dfdx_p(X1<0)=0;
            
            dfdx_n(X1>0)=0;
            
            dfdx =dfdx_p + dfdx_n;
            
           dfda_p = -X.*(layer.b1./(2*(layer.a.*X + 1))...
               + (X.*(layer.b1 - layer.b1.*layer.a))./(2*(layer.a.*X + 1).^2)...
               + (layer.b1.*(X + 1))./(2*(layer.a - X + layer.a.*X).^2));

           dfda_n = X.*(layer.b2./(2*(layer.a.*X + 1))...
               + (X.*(layer.b2 - layer.b2.*layer.a))./(2*(layer.a.*X + 1).^2)...
               + (layer.b2.*(X + 1))./(2*(layer.a - X + layer.a.*X).^2));
           
           dfda_p(X1<0)=0;
            
           dfda_n(X1>0)=0;
            
           dfda =dfda_p + dfda_n;
           
           dfdb1 = -X.*((layer.a - 1)./(2*(layer.a.*X + 1))...
               - 1./(2*(layer.a - X + layer.a.*X)));
           
           dfdb2 = X.*((layer.a - 1)./(2*(layer.a.*X + 1))...
               - 1./(2*(layer.a - X + layer.a.*X)));
          
           dfdb1(X1<0)=0;
           dfdb2(X1>0)=0;
end