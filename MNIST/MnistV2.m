clc;clear;close all;

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% figure;
% perm = randperm(10000,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(imds.Files{perm(i)});
% end

labelCount = countEachLabel(imds)
img = readimage(imds,1);
size(img)

numTrainFiles = 700;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
global scale
scale=1;
layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3,8,'Padding','same')
%     batchNormalizationLayer
    SIT2FMLayerOpt(8)
%     lereluLayer(8)
%     eluLayer(8)
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
    
%     convolution2dLayer(3,16,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
    
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
% vFreq = floor(numel(imdsTrain.Labels)/128);
options=trainingOptions('sgdm', ...
    'InitialLearnRate', 10^-3, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency', 15, ...
    'MaxEpochs', 15, ...
    'MiniBatchSize',128, ...
    'Verbose', true,...
    'Plots','training-progress');

% options = trainingOptions('sgdm', ...
%     'InitialLearnRate',0.001, ...
%     'MaxEpochs',10, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',imdsValidation, ...
%     'ValidationFrequency',30, ... 
%     'Verbose',true);

[net, info] = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)