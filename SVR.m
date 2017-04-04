load allstate;

trainX = X(1:131822,: ,:);
trainY = Y(1:131822,: ,:);
valX = X(131823:150653, : ,:); %18831
valY = Y(131823:150653, : ,:); %18831
testX = X(150654:end, :, :);  %38663 
testY = Y(150654:end, :, :);  %38663

x = trainX;
y= trainY; 
kernelFunction = 'gaussian';

tic; 
svmrModel = fitrsvm(x, y,  'KernelFunction', kernelFunction, 'Standardize', true, 'KFold', 5);
toc; 

%%%%%%%%%%%%%%% FOR PARTITIONED MODELS
  mse = mean(kfoldLoss(svmrModel));
  Rmse = min(sqrt(mse)); 
  
numCLModels = numel(svmrModel.Trained);
Rsquares = 0;
Rmses =0;
Mapes =0;
for i=1:numCLModels
    mdl = svmrModel.Trained{i};
    Yfit = predict(mdl, testX);    
    yresid = testY - Yfit;
    SSresid = sum(yresid.^2);
    SStotal = (length(testY)- 1) * var(testY);
    rsquare =mean(1 - min(SSresid)/SStotal);
    mse = min(mean(yresid.^2));
    Rmse = sqrt(mse); 
    mape = mean(abs(yresid./testY)) *100;     
    Rsquares = Rsquares + rsquare;   
    Rmses = Rmses + min(Rmse);
    Mapes  = Mapes + min(mape);
end 
    Rsquares = Rsquares/numCLModels; 
    Rmses = Rmses /numCLModels;
    Mapes  = Mapes /numCLModels; 
 


