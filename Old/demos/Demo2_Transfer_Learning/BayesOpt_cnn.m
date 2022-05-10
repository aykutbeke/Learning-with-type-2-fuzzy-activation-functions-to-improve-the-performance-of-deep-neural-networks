function ObjFcn = BayesOpt_cnn(train_DS, val_DS, network)
% Copyright 2018 The MathWorks, Inc.
ObjFcn = @valErrorFun1;

    function [valError] = valErrorFun1(optVars)
        %% CNN Architecture
        net = eval(network);
        layers = net.Layers;        
        
        switch network
            case 'alexnet'
                %% Alter network to fit our desired output
                % The pre-trained layers at the end of the network are designed to classify
                % 1000 objects. But we need to classify different objects now. So the
                % first step in transfer learning is to replace alter just two of the layers of the
                % pre-trained network with a set of layers that can classify 5 classes.
                
                % Get the layers from the network. The layers define the network
                % architecture and contain the learned weights. Here we only alter two of
                % the layers. Everything else stays the same.
                
                num_objects = height(train_DS.countEachLabel);
                
                layers(end-2) = fullyConnectedLayer(num_objects, 'Name','fc8');
                layers(end) = classificationLayer('Name','myNewClassifier');
                
                layers_train = layers;
            case 'googlenet'
                % Extract the layer graph from the trained network and plot the layer
                % graph.
                lgraph = layerGraph(net);
                 
                %%
                % To retrain GoogLeNet to classify new images, replace the last three
                % layers of the network. These three layers of the network, with the names
                % |'loss3-classifier'|, |'prob'|, and |'output'|, contain the information
                % of how to combine the features that the network extracts into class
                % probabilities and labels. Add three new layers, a fully connected layer,
                % a softmax layer, and a classification output layer, to the layer graph.
                % Set the final fully connected layer to have the same size as the number
                % of classes in the new data set (5, in this example). To learn faster in
                % the new layers than in the transferred layers, increase the learning rate
                % factors of the fully connected layer.
                lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});
                
                numClasses = numel(categories(train_DS.Labels));
                newLayers = [
                    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor', 20)
                    softmaxLayer('Name','softmax')
                    classificationLayer('Name','classoutput')];
                lgraph = addLayers(lgraph,newLayers);
                
                
                %%
                % Connect the last of the transferred layers remaining in the network
                % (|'pool5-drop_7x7_s1'|) to the new layers. To check that the new layers
                % are correctly connected, plot the new layer graph and zoom in on the last
                % layers of the network.
                lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');
                
                layers_train = lgraph;
        end
        
        options = trainingOptions('sgdm',...
            'ValidationData',val_DS,...
            'InitialLearnRate',optVars.InitialLearnRate,...,
            'Plots','training-progress', ...
            'MiniBatchSize', 64, ...
            'ValidationPatience', 3);
            % 'LearnRateSchedule','piecewise',...
            % 
            % 'LearnRateDropFactor',optVars.LearnRateDropFactor,...
            % 'LearnRateDropPeriod',optVars.LearnRateDropPeriod,...
            % 'Momentum',optVars.Momentum,...
            % 'MaxEpochs',optVars.MaxEpochs,...
            % 'MiniBatchSize',optVars.MiniBatchSize,... 
        
        %% Train the network
        net = trainNetwork(train_DS, layers_train, options);
        %% Validation accuracy
        [labels,~] = classify(net, val_DS, 'MiniBatchSize', 64);
        accuracy_training  = sum(labels== val_DS.Labels )./numel(labels);
        %plot(accuracy_training); hold on;
        valError = 1 - accuracy_training;
        
    end % end for inner function
end % end for outer function






