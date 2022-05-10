clear; clc 
%% Load Data
load drydemodata
data = y2';
output = y2';
input=u2';

% data = chickenpox_dataset;
% data = [data{:}];

%% Divide Data: Training and Testing
numTimeStepsTrain = floor(0.85*numel(data));
% XTrain = [data(1:numTimeStepsTrain-1);data(2:numTimeStepsTrain)];
% YTrain = data(3:numTimeStepsTrain+1);
% XTest = [data(numTimeStepsTrain+1:end-2);data(numTimeStepsTrain+2:end-1)];
% YTest = data(numTimeStepsTrain+3:end);
XTrain = [input(1:numTimeStepsTrain-1);output(1:numTimeStepsTrain-1)];
YTrain = output(2:numTimeStepsTrain);
XTest = [input(numTimeStepsTrain:end-1);output(numTimeStepsTrain:end-1)];
YTest = output(numTimeStepsTrain+1:end);


%% Standardize Data
mu_x = mean(XTrain')'; mu_y = mean(YTrain')';
sig_x = std(XTrain')'; sig_y = std(YTrain')';
XTrain = (XTrain - mu_x) ./ sig_x;
YTrain = (YTrain - mu_y) ./ sig_y;
XTest = (XTest - mu_x) ./ sig_x;
%% Define LSTM Network
inputSize = 2;
numResponses = 1;
numHiddenUnits = 500;
layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
%% Training Options
opts = trainingOptions('adam', ...
    'MaxEpochs',500, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
%% Train Network
net = trainNetwork(XTrain,YTrain,layers,opts);
%% Forecast Future Time Steps
% net = predictAndUpdateState(net,XTrain);
% [net,YPred(1,1)] = predictAndUpdateState(net,[XTrain(end-2);XTrain(end-1)]);
% [net,YPred(1,2)] = predictAndUpdateState(net,[XTrain(end-1);XTrain(end)]);
% numTimeStepsTest = size(XTest,2);
% for i = 3:numTimeStepsTest
%     [net,YPred(1,i)] = predictAndUpdateState(net,[XTest(i-2);XTest(i-1)]);
% end

net = predictAndUpdateState(net,XTrain);
% [net,YPred(1,1)] = predictAndUpdateState(net,[XTrain(end-2);XTrain(end-1)]);
numTimeStepsTest = size(XTest,2);
for i = 1:numTimeStepsTest
    [net,YPred(1,i)] = predictAndUpdateState(net,[XTest(:,i)]);
end
%% Unstandardize the predictions using mu and sig calculated earlier.
YPred = sig_y*YPred + mu_y;
%% RMSE and MAE Calculation
rmse = sqrt(mean((YPred-YTest).^2))
MAE = mae(YPred-YTest)
%% Plot results
figure
plot(data(1:numTimeStepsTrain))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Month")
ylabel("Cases")
title("Forecast")
legend(["Observed" "Forecast"])
%% Compare the forecasted values with the test data
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")
subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)