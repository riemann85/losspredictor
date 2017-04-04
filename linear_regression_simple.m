load allstate;

trainX = X(1:131822,: ,:);
trainY = Y(1:131822,: ,:);
valX = X(131823:150653, : ,:); %18831
valY = Y(131823:150653, : ,:); %18831
testX = X(150654:end, :, :);  %38663 
testY = Y(150654:end, :, :);  %38663

x = trainX;
y= trainY;
 
%Using only categorical variable, traing the model using linear regression 
tic;
lrmodel = fitlm(x, y);
toc;

% %%%%%%%%%%%%%%% FOR SINGLE MODELS
predictedY =  predict(lrmodel, testX);
yresid = testY -predictedY ;
yresid(isnan(yresid)) = 0; 
SSresid = sum(yresid.^2);
SStotal = (length(testY)- 1) * var(testY);
rsquare = 1 - SSresid/SStotal;
mse = mean(yresid.^2);
Rmse = sqrt(mse); 
mape = mean(abs(yresid./testY)) *100 ; 
