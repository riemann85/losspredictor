
load allstate;

trainX = X(1:131822,: ,:);
trainY = Y(1:131822,: ,:);
valX = X(131823:150653, : ,:); %18831
valY = Y(131823:150653, : ,:); %18831
testX = X(150654:end, :, :);  %38663 
testY = Y(150654:end, :, :);  %38663
 

x = trainX';
y= trainY;

[B, fitInfo] = lasso(x, y, 'CV', 10);  %Construct the lasso fit using ten-fold cross- validation.
lassoPlot(B, fitInfo, 'PlotType', 'CV');
bestValue = find(fitInfo.Lambda == fitInfo.LambdaMinMSE);
lambda = fitInfo.Lambda;
minpts = find(B(:,fitInfo.IndexMinMSE));
min1pts = find(B(:,fitInfo.Index1SE));
B(min1pts,fitInfo.Index1SE);