%% Deep learning for classification on the MNIST dataset
% Copyright 2018 The MathWorks, Inc.
clc;clear;close all
%% Prepare the dataset

% The MNIST dataset is a set of handwritten digits categorized 0-9 and is
% available at http://yann.lecun.com/exdb/mnist/.

% The following line will download (if necessary) and prepare the dataset
% to use in MATLAB.
[imgDataTrain, labelsTrain, imgDataTest, labelsTest] = prepareData;

%% Let's look at a few of the images
% Randomize the images for display
warning off images:imshow:magnificationMustBeFitForDockedFigure
perm = randperm(numel(labelsTrain), 25);
subset = imgDataTrain(:,:,1,perm);
montage(subset)

%% How do we classify a digit?
% First, we need a model - let's load one
load MNISTModel

% Predict the class of an image
randIndx = randi(numel(labelsTest));
img = imgDataTest(:,:,1,randIndx);
actualLabel = labelsTest(randIndx);

% predictedLabel = net.classify(img);
% imshow(img);
% title(['Predicted: ' char(predictedLabel) ', Actual: ' char(actualLabel)])

%% Need a starting point? Check the documentation!
% search "deep learning"
% web(fullfile(docroot, 'nnet/deep-learning-training-from-scratch.html'))


%% Prepare the CNN
% One of the simplest possible convnets, it contains one convolutional
% layer, one ReLU, one pooling layer, and one fully connected layer

layers = [  imageInputLayer([28 28 1])
            convolution2dLayer(5,20)
            batchNormalizationLayer
%             reluLayer
%             eluLayer(20)
            SIT2FMLayer(20)
%             maxPooling2dLayer(2, 'Stride', 2)
            fullyConnectedLayer(10)
            softmaxLayer
            classificationLayer()   ]
        
%% Attempt 1: Set training options and train the network

    
    miniBatchSize = 128;
    options = trainingOptions( 'sgdm',...
        'MiniBatchSize', miniBatchSize,...
        'InitialLearnRate', 0.0001,...
        'MaxEpochs',5,...
        'Verbose', true);
%     'Plots', 'training-progress'

    [net, info] = trainNetwork(imgDataTrain, labelsTrain, layers, options);
    

predLabelsTest = net.classify(imgDataTest);
testAccuracy = sum(predLabelsTest == labelsTest) / numel(labelsTest)
%% Attempt 2: Change the learning rate


    options = trainingOptions( 'sgdm',...
        'MiniBatchSize', miniBatchSize,...
        'Plots', 'training-progress',...
        'InitialLearnRate', 0.0001);

    net = trainNetwork(imgDataTrain, labelsTrain, layers, options);

%% Attempt 3: Change the network architecture

layers = [
    imageInputLayer([28 28 1])

    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

    options = trainingOptions( 'sgdm',...
        'MiniBatchSize', miniBatchSize,...
        'Plots', 'training-progress');

    net = trainNetwork(imgDataTrain, labelsTrain, layers, options);
    


%% Classify the test data set

predLabelsTest = net.classify(imgDataTest);

testAccuracy = sum(predLabelsTest == labelsTest) / numel(labelsTest)

%% Try to classify something else
img = imread('letterW.png');
actualLabel = 'W';

predictedLabel = net.classify(img);
imshow(img);
title(['Predicted: ' char(predictedLabel) ', Actual: ' char(actualLabel)])