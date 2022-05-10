function Z = predict(X,a1,b1,b2)
            eps = 0; 
            layer.a = a1;
            layer.b1 =b1;
            layer.b2 = b2;
%             layer.b2 = ones(1,1,20)*0.0;
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
%             layer.a1(layer.a1>0.95) = 0.95 ; layer.a1(layer.a1<0.05) = 0.05;
%             layer.a2(layer.a2>0.95) = 0.95 ; layer.a2(layer.a2<0.05) = 0.05;
%             layer.b1(layer.b1>0.95) = 0.95 ; layer.b1(layer.b1<0.05) = 0.05;
%             layer.b2(layer.b2>0.95) = 0.95 ; layer.b2(layer.b2<0.05) = 0.05;
            
            Xp = max(0,X);
            Xn = min(0,X);
            X = abs(X);
            Kp = 0.5*(layer.b1./(X - X.*layer.a + layer.a)...
                    - (layer.b1-layer.b1.*layer.a)./(X.*layer.a-1));
            
            Kn = 0.5*(layer.b2./(X - X.*layer.a + layer.a)...
                    - (layer.b2-layer.b2.*layer.a)./(X.*layer.a-1));
            
            Z = Xp.*Kp + Xn .*Kn;
end