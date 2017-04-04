load allstate;

x = X';
t = Y';

trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize,trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 10/100;

[net,tr] = train(net,x,t);
y = net(x);
error = gsubtract(t,y);
performance = perform(net,t,y);

% View the Network
view(net)
mse = mean(error.^2);
Rmse = sqrt(mse);
SSresid = sum(error.^2);
SStotal = (length(Y)- 1) * var(Y);
rsquare = 1 - SSresid/SStotal;
mape = mean(abs(error./Y')) *100 ;

plotperform(tr)
plottrainstate(tr)
ploterrhist(e)
plotregression(t,y)
plotfit(net,x,t)