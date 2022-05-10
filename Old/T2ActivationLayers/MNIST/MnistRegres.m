clc, clear, close all;
[XTrain,~,YTrain] = digitTrain4DArrayData;
[XValidation,~,YValidation] = digitTest4DArrayData;

XTrain = cat(4, XTrain, XValidation(:,:,:,1:2000));
YTrain = [YTrain; YValidation(1:2000)];
XValidation = XValidation(:,:,:,2001:end);
YValidation = YValidation(2001:end);
% numTrainImages = numel(YTrain);

% figure
% idx = randperm(numTrainImages,20);
% for i = 1:numel(idx)
%     subplot(4,5,i)    
%     imshow(XTrain(:,:,:,idx(i)))
%     drawnow
% end
% 
% figure
% histogram(YTrain)
% axis tight
% ylabel('Counts')
% xlabel('Rotation Angle')
global scale
scale=1;
layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
%     eluLayer(32)
%       lereluLayer(32)
%     SIT2FMLayerOpt(32)   
     reluLayer
%     
%     averagePooling2dLayer(2,'Stride',2)
% 
%     convolution2dLayer(3,16,'Padding','same')
%     batchNormalizationLayer
% %     reluLayer
%     SIT2FMLayer(16)
%     
%     averagePooling2dLayer(2,'Stride',2)
%   
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
% %     reluLayer
%     SIT2FMLayer(32)
%     
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
% %     reluLayer
%     SIT2FMLayer(32)
    
%     dropoutLayer(0.2)
    fullyConnectedLayer(1)
    regressionLayer];

miniBatchSize  = 128;
% validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',15, ...
    'InitialLearnRate',1e-3, ...
    'Plots','training-progress', ...
    'LearnRateSchedule','piecewise', ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation, YValidation}, ...
    'ValidationFrequency',15, ...
    'Verbose',true);

[net, info] = trainNetwork(XTrain,YTrain,layers,options);

YPredicted = predict(net,XValidation);

predictionError = YValidation - YPredicted;
thr = 10;
numCorrect = sum(abs(predictionError) < thr);
numValidationImages = numel(YValidation);

accuracy = numCorrect/numValidationImages
squares = predictionError.^2;
rmse = sqrt(mean(squares))