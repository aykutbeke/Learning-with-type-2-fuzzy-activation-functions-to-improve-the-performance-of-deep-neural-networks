clc;clear;h= findall(groot,'Type','Figure');close(h)
% rng('default')
%% Load
load('icEngine')
% y2=y(1:end);
% u2=u(1:end);
% sy =1; su=1;
index = 1000;
[y2_tr,sy_tr] = nor(y(1:index));
[u2_tr,su_tr] = nor(u(1:index));
[y2_te,sy_te] = nor(y(index+1:end));
[u2_te,su_te] = nor(u(index+1:end));
y2=[y2_tr;y2_te];u2=[u2_tr;u2_te];
%% Data preperation 
inputRegressor = prepareRegressor(y2,u2,[{'y(k-1), y(k-2), y(k-3), y(k-4), u(k-10), u(k-11)'}]); 
subplot(2,1,1)
plot(1:index,y2(1:index),'-',1:index,y2(1:index),'o')
ylabel('y(k)','fontsize',10)
subplot(2,1,2)
plot(1:index,u2(1:index),'-',1:index,u2(1:index),'o')
ylabel('u(k)','fontsize',10)
%%  FFNetwork
global Kf
Kf=1;
prediction=1;
FF_NetworkLayers = [ ...
    sequenceInputLayer(size(inputRegressor{1}',1),'Name','FF-Giris')
%     fullyConnectedLayer(60,'Name','RegressionLayer')
%     sequenceFoldingLayer
%     fullyConnectedLayer(6,'Name','FF-Output2')
%     purelinLay('PureFF_2')    
    SIT2FRU(6, 'SIT2')
    fullyConnectedLayer(1,'Name','FF-Output')
%     purelinLay('PureFF_1')
%     SIT2FRU(1, 'SIT22')
    regressionLayer];

[trainInd,valInd] = divideint(size(inputRegressor{1}',2)-prediction,1,0,0);  %% -1 dikkat
% ValidationInput=tonndata(Fk_Train(:,valInd),true,false); 
% ValidationTarget=tonndata(Fk__pluts_1_Train(:,valInd),true,false); 
global Kf
Kf=1;
FF_NetworkOptions = trainingOptions('sgdm', ...     'ValidationData',{ValidationInput,ValidationTarget},...
    'MaxEpochs',1500, ...
    'GradientThreshold',5, ...
    'GradientThresholdMethod','l2norm',...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',500, ...
    'LearnRateDropFactor',0.5, ...
    'MiniBatchSize',100,...
    'Shuffle','never',...
    'L2Regularization',0.04,... 
    'Plots','training-progress'); %% 0.083   NN  : 0.044

FF_Network= trainNetwork(inputRegressor{1}(1:index,:)', y2(1:index,:)',FF_NetworkLayers,FF_NetworkOptions);

PredFF_train = predict(FF_Network, inputRegressor{1}(1:index,:)');
PredFF_test = predict(FF_Network, inputRegressor{1}(index+1:end,:)');

PredFF_train = PredFF_train*sy_tr;
PredFF_test = PredFF_test*sy_te;

figure
plot(y(1:index))
hold on
plot(PredFF_train)
ac = 100*(1-norm(y(1:index)-double(PredFF_train)')/norm(y(1:index)-mean(y(1:index))));
title('Accuracy: '+string(ac))
figure
plot(y(index+1:end))
hold on
plot(PredFF_test)
ac = 100*(1-norm(y(index+1:end)-double(PredFF_test)')/norm(y(index+1:end)-mean(y(index+1:end))));
title('Accuracy: '+string(ac))

plotT2Layers(FF_NetworkLayers, FF_Network)



