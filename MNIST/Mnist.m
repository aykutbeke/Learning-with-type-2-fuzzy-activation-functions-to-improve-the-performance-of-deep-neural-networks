clear;close all;clc
%% Old 
[XTrain,YTrain] = digitTrain4DArrayData;

layers = [ 
    imageInputLayer([28 28 1])
    convolution2dLayer(5,20)
%     batchNormalizationLayer
%     NormalizationLayer
%     reluLayer
%       lereluLayer(20)
%     eluLayer(20)
    SIT2FMLayerOpt(20)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer]
% miniBatchSize = 100;
% options = trainingOptions( 'sgdm',...
%     'MiniBatchSize', miniBatchSize,...
%     'Plots', 'training-progress');
% rng('default')
options = trainingOptions('adam','MaxEpochs',15,'InitialLearnRate', 10^-2);
net = trainNetwork(XTrain,YTrain,layers,options);

[XTest,YTest] = digitTest4DArrayData;
YPred = classify(net,XTest);
accuracy = sum(YTest==YPred)/numel(YTest)
%% Deep learning for classification on the MNIST dataset
%% Test Data
% Extract the first convolutional layer weights
w = net.Layers(2).Weights;
a1 = net.Layers(3).a1;
a2 = net.Layers(3).a2;
b1 = net.Layers(3).b1;
b2 = net.Layers(3).b2;
% rescale the weights to the range [0, 1] for better visualization
w = rescale(w);
% 
% figure
% montage(w)
% figure
% montage(a)
figure
plot(a1(:,:),'*')
figure
plot(a2(:,:),'*')
figure
plot(b1(:,:),'*')
figure
plot(b2(:,:),'*')