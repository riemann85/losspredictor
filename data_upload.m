
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%       DATA PREPROCESSING    %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Import the output dataset to matlab (The values to be predicted)
Y = xlsread('totalY.xlsx','Sheet1','1:188319'); 

%Code to Normalize 
normY = normc(totalY); 

%Import the continuous input(response variable) dataset to matlab (The values to be predicted)
contX = xlsread('totalContX.xlsx','Sheet1','1:188318');  
 
%Import the continuous input(response variable) dataset to matlab (The values to be predicted)
totalCatXint = xlsread('totalCatXint.xlsx','Sheet1','1:188318'); 
totalCatXint2 = xlsread('totalCatXint2.xlsx','Sheet1','1:188318'); 
catX = horzcat (totalCatXint, totalCatXint2);

%concatenate both categorical and continuous variable
X = horzcat (catX, contX);

%to formulate the total dataset will be X=predictor variable  ; Y = response variable 
totalData = horzcat (X, Y);