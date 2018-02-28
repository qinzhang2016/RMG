%Demo for RMG code

%load dataest

%ProteinData is the original dataset
%proteinTrn is a random partition of ProteinData and scaled to [-1,+1]
%trX and trLab are the labeled data and corresponding labels;
%tsX and tsLab are the unlabeled data and corresponding labels;
%indTrn and indTst are the corresponding index in the original dataset
%
load('ProteinData');
load('proteinTrn'); 

[N,Dim] = size(data);
X=NewScale(double(data));

%parameters
kg=100;
kf=floor(log2(Dim)+1);

tic;
[G,F]=MultiGraphs(X,labels,indTrn,kg,kf);
time = toc;

%compute the label
[a,y]=max(F,[],2);

acc = ComputeAcc(y,labels,size(trX,1))
runTime = time