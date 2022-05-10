function accuracy = validatePerformance(net,testDS,doTest)
% Copyright 2018 The MathWorks, Inc.
%% Test new classifier on validation set
% Now run the network on the test data set to see how well it does
% Please note: this will take a while to run.

if doTest
    imagepath = fullfile('foodData', 'test');
    tic
    [labels,err_test] = classify(net, testDS);
    toc
else % all of the test images have been classified and saved in a mat file:
    load(fullfile('results','test2.mat'));
end

accuracy = sum(labels == testDS.Labels)/numel(labels);
disp(['Test accuracy is ' num2str(accuracy)])

% Somewhat easy confusion matrix - heat map (Base MATLAB - new in 17A)
tt = table(labels, testDS.Labels,'VariableNames',{'Predicted','Actual'});
figure('name','confusion matrix'); heatmap(tt,'Actual','Predicted');
pause(1)

% Or create a more 'sophisticated' confusion matrix
tbl = testDS.countEachLabel;
nl = length(unique(testDS.Labels));
t = zeros(nl,length(labels));
y = t;
for ii = 1:nl
    y(ii,:) = labels == tbl.Label(ii);
    t(ii,:) = testDS.Labels == tbl.Label(ii);
end
figure;
plotconfusion(t,y);
