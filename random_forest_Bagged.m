load allstate;

trainX = X(1:131822,: ,:);
trainY = Y(1:131822,: ,:);
valX = X(131823:150653, : ,:); %18831
valY = Y(131823:150653, : ,:); %18831
testX = X(150654:end, :, :);  %38663 
testY = Y(150654:end, :, :);  %38663 

x = trainX;
y= trainY;

tic; 
t = RegressionTree.template('MinLeaf',10);
bagModel = TreeBagger(100,x,y, 'Method', 'regression'); 
toc;  

predictedY =  predict(bagModel, testX);
yresid = testY -predictedY ;  
SSresid = sum(yresid.^2);
SStotal = (length(testY)- 1) * var(testY);
rsquare = 1 - SSresid/SStotal;
mse = mean(yresid.^2);
Rmse = sqrt(mse); 
mape = mean(abs(yresid./testY)) *100 ; 
err = error (bagModel, testX, testY, 'Mode', 'cumulative');

figure(1); 
plot(testY,'b','LineWidth',1), hold on
plot( predictedY, 'r.-', 'LineWidth',1)

% Observe hundred points, pan to view more
xlim([0 100])
legend({'Actual','Predicted'})
xlabel('Test Data Points');
ylabel('Claim Severity (Loss)'); 

mTestX = mean(testX, 2); 
figure(2)
scatter( mTestX, testY,'b','.'), hold on
scatter(mTestX, predictedY, 'r','.')
xlabel('Loss')
ylabel('Test Data Points');
title('Actual Vs Predicted')
legend('Actual Loss', 'Predicted loss') 
 

figure(3);
plot3(bagModel.X,bagModel.Y, bagModel.W ,'.');
xlabel('tbaggerModel.X'); ylabel('tbaggerModel.Y'); zlabel('tbaggerModel.W');
title('Tree Baggger Model') 

save('treebagger' , 'bagModel', 'rsquare', 'SSresid', 'SStotal', 'yresid', 'err')

