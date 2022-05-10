
        function Z = nor(X)
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
            X_p = max(0, X);
            X_n = min(0, X);
            scale = max(max(X_p),max(abs(X_n)));
            
            Z = X_p./(scale+eps) + X_n./(scale+eps);
           
        end