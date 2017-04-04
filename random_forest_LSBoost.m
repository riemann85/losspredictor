
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
model = fitensemble(x,y,'LSBoost',1000,t,'LearnRate',1.00);  
toc;
L = loss(model,x,y,'mode','ensemble');
fprintf('Mean-square testing error = %f\n',L);
 
predictedY =  predict(model, testX);
yresid = testY -predictedY ;  
SSresid = sum(yresid.^2);
SStotal = (length(testY)- 1) * var(testY);
rsquare = 1 - SSresid/SStotal;
mse = mean(yresid.^2);
Rmse = sqrt(mse); 
mape = mean(abs(yresid./testY)) *100 ; 

figure(1); 
plot(testY,'b','LineWidth',2), hold on
plot( predictedY, 'r.-', 'LineWidth',1,'MarkerSize',15)

% Observe first hundred points, pan to view more
xlim([0 100])
legend({'Actual','Predicted'})
xlabel('Test Data Points');
ylabel('Claim Severity (Loss)');

[predictorImportance,sortedIndex] = model.predictorImportance;
figure(2);
barh(predictorImportance)
set(gca,'ytickLabel',X(sortedIndex))
view(model.Trained{1},'Mode','graph');
xlabel('Predictor Importance')