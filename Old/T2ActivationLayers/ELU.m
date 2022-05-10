alpha1 = 1;
elu_fcn = @(x) x.*(x > 0) + alpha1*(exp(x) - 1).*(x <= 0);

alpha2 = 0.1;
leaky_relu_fcn = @(x) alpha2*x.*(x <= 0) + x.*(x > 0);

relu_fcn = @(x) x.*(x > 0);

fplot(elu_fcn,[-10 3],'LineWidth',2)
hold on
fplot(leaky_relu_fcn,[-10 3],'LineWidth',2)
fplot(relu_fcn,[-10 3],'LineWidth',2)
hold off
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
box off
legend({'ELU','Leaky ReLU','ReLU'},'Location','northwest')
%% mnist Example With ELU
clc; clear; close all
[XTrain, YTrain] = digitTrain4DArrayData;
imshow(XTrain(:,:,:,1010),'InitialMagnification','fit')
YTrain(1010)
%% Make a network that uses our new ELU layer.
layers = [ ...
    imageInputLayer([28 28 1])
    convolution2dLayer(5,20)
%     batchNormalizationLayer
%     defaultLayer(10,10)
    examplePreluLayer(20)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
%% Train the network.
options = trainingOptions('sgdm');
net = trainNetwork(XTrain,YTrain,layers,options);
%% Check the accuracy of the network on our test set.
[XTest, YTest] = digitTest4DArrayData;
YPred = classify(net, XTest);
accuracy = sum(YTest==YPred)/numel(YTest)
%% Look at one of the images in the test set and see how it was classified by the network.
k = 1500;
imshow(XTest(:,:,:,k),'InitialMagnification','fit')
YPred(k)