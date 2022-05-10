%% Load Data
data = chickenpox_dataset;
data = [data{:}];

%% Divide Data: Training and Testing
numTimeStepsTrain = floor(0.85*numel(data));
XTrain = data(1:numTimeStepsTrain);
YTrain = data(2:numTimeStepsTrain+1);
XTest = data(numTimeStepsTrain+1:end-1);
YTest = data(numTimeStepsTrain+2:end);
%% Standardize Data
mu = mean(XTrain);
sig = std(XTrain);
XTrain = (XTrain - mu) / sig;
YTrain = (YTrain - mu) / sig;
XTest = (XTest - mu) / sig;
%% Define LSTM Network
inputSize = 1;
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
    'Plots','training-progress','ExecutionEnvironment','gpu');
%% Train Network
net = trainNetwork(XTrain,YTrain,layers,opts);
%% Forecast Future Time Steps
net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));
numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(1,i)] = predictAndUpdateState(net,YPred(i-1));
end
%% Unstandardize the predictions using mu and sig calculated earlier.
YPred = sig*YPred + mu;
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