clear;close all;clc
%% Old 
[XTrain,YTrain] = digitTrain4DArrayData;

layers = [ 
    imageInputLayer([28 28 1])
    convolution2dLayer(5,20)
    batchNormalizationLayer
%     NormalizationLayer
%     reluLayer
%     preluLayer(20)
%     eluLayer(20)
    singleInputT2Layer(20)
%     SIT2FMLayer(20)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer]
miniBatchSize = 900;
% options = trainingOptions( 'sgdm',...
%     'MiniBatchSize', miniBatchSize,...
%     'Plots', 'training-progress');
rng('default')
options = trainingOptions('adam','MaxEpochs',30, 'MiniBatchSize', miniBatchSize,'InitialLearnRate', 10^-2);
net = trainNetwork(XTrain,YTrain,layers,options);

[XTest,YTest] = digitTest4DArrayData;
YPred = classify(net,XTest);
accuracy = sum(YTest==YPred)/numel(YTest)
%% Deep learning for classification on the MNIST dataset
% [imgDataTrain, labelsTrain, imgDataTest, labelsTest] = preparneData;
% warning off images:imshow:magnificationMustBeFitForDockedFigure
% perm = randperm(numel(labelsTrain), 25);
% subset = imgDataTrain(:,:,1,perm);
% montage(subset)