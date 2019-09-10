from pandas import *
from sklearn.preprocessing import LabelEncoder
from numpy import *
import matplotlib.pyplot as plt

def findRmse(err,n):
	return ((1/n)*dot(err.transpose(),err))**0.5

def sigmoid(x,theta):
	z=dot(x,theta)
	return 1/(1+exp(-z))

def gradientDescent(xTrain,yTrain,theta,learningRate,iterations):
	rmse=[]
	n=len(xTrain)
	for i in range(iterations):
		hypo=sigmoid(xTrain,theta)
		err=hypo-yTrain
		rmse.append(findRmse(err,len(err))[0][0])
		theta=theta-(learningRate*(1/n)*dot(xTrain.transpose(),err))
	return theta,rmse

def gradientDescentWithL1(xTrain,yTrain,theta,learningRate,iterations,alpha):
	rmse=[]
	n=len(xTrain)
	for i in range(iterations):
		hypo=sigmoid(xTrain,theta)
		err=hypo-yTrain
		rmse.append(findRmse(err,len(err))[0][0])
		theta=theta-(learningRate*((1/n)*dot(xTrain.transpose(),err)))-((learningRate*alpha*sign(theta))/n)
	return theta,rmse

def gradientDescentWithL2(xTrain,yTrain,theta,learningRate,iterations,alpha):
	rmse=[]
	n=len(xTrain)
	for i in range(iterations):
		hypo=sigmoid(xTrain,theta)
		err=hypo-yTrain
		rmse.append(findRmse(err,len(err))[0][0])
		theta=theta*(1-(alpha/n))-((learningRate/n)*(dot(xTrain.transpose(),err)))
	return theta,rmse

def run():
	data1=read_csv("train.csv",names=["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","output"])
	xTrain=data1.iloc[:,0:14]
	yTrain=data1.iloc[:,14].values

	data1=read_csv("test.csv",names=["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","output"])
	xTest=data1.iloc[:,0:14]
	yTest=data1.iloc[:,14].values

	x=concat([xTrain,xTest])
	x=get_dummies(x,prefix_sep='_',drop_first=True)
	xPart1=x.iloc[:,0:6]
	xPart2=x.iloc[:,6:,]
	xPart1=(xPart1-xPart1.mean())/xPart1.std()
	x=concat([xPart1,xPart2],axis=1)
	xTrain=x.iloc[0:30162,:]
	xTest=x.iloc[30162:,:]
	xTrain=append(ones((xTrain.shape[0],1)),xTrain,axis=1)
	xTest=append(ones((xTest.shape[0],1)),xTest,axis=1)


	le=LabelEncoder()
	yTrain=le.fit_transform(yTrain)
	yTest=le.fit_transform(yTest)
	yTrain=yTrain.reshape((yTrain.shape[0],1))
	yTest=yTest.reshape((yTest.shape[0],1))
	xVal=xTrain[int(0.8*len(xTrain)):]
	yVal=yTrain[int(0.8*len(yTrain)):]
	xTrain=xTrain[:int(0.8*len(xTrain))]
	yTrain=yTrain[:int(0.8*len(yTrain))]

	iterations=5000
	learningRate=0.01
	alpha=0.1
	theta=zeros((xTrain.shape[1],1))
	theta,rmse3=gradientDescent(xTrain,yTrain,theta,learningRate,iterations)

	theta1=zeros((xTrain.shape[1],1))
	theta1,rmse1=gradientDescentWithL1(xTrain,yTrain,theta1,learningRate,iterations,alpha)

	theta2=zeros((xTrain.shape[1],1))
	theta2,rmse2=gradientDescentWithL2(xTrain,yTrain,theta2,learningRate,iterations,alpha)

	pred1Test=sigmoid(xTest,theta)
	pred1Val=sigmoid(xVal,theta)
	for i in range(len(pred1Val)):
		if pred1Val[i,0]>0.5:
			pred1Val[i,0]=1
		else:
			pred1Val[i,0]=0
	count=0
	for i in range(len(pred1Val)):
		if yVal[i]==pred1Val[i]:
			count+=1
	print("Accuracy on validation set without regularisation:",(count/len(pred1Val))*100,"%")

	for i in range(len(pred1Test)):
		if pred1Test[i,0]>0.5:
			pred1Test[i,0]=1
		else:
			pred1Test[i,0]=0
	count=0
	for i in range(len(pred1Test)):
		if yTest[i]==pred1Test[i]:
			count+=1
	print("Accuracy on test set without regularisation:",(count/len(pred1Test))*100,"%")
	print()


	pred2Train=sigmoid(xTrain,theta1)
	for i in range(len(pred2Train)):
		if pred2Train[i,0]>0.5:
			pred2Train[i,0]=1
		else:
			pred2Train[i,0]=0
	count=0
	for i in range(len(pred2Train)):
		if yTrain[i]==pred2Train[i]:
			count+=1
	print("Accuracy on training set with L1 regularisation:",(count/len(pred2Train))*100,"%")

	pred2Val=sigmoid(xVal,theta1)
	for i in range(len(pred2Val)):
		if pred2Val[i,0]>0.5:
			pred2Val[i,0]=1
		else:
			pred2Val[i,0]=0
	count=0
	for i in range(len(pred2Val)):
		if yTest[i]==pred2Val[i]:
			count+=1
	print("Accuracy on validation set with L1 regularisation:",(count/len(pred2Val))*100,"%")

	pred2Test=sigmoid(xTest,theta1)
	for i in range(len(pred2Test)):
		if pred2Test[i,0]>0.5:
			pred2Test[i,0]=1
		else:
			pred2Test[i,0]=0
	count=0
	for i in range(len(pred2Test)):
		if yTest[i]==pred2Test[i]:
			count+=1
	print("Accuracy on test set with L1 regularisation:",(count/len(pred2Test))*100,"%")
	print()

	pred3Train=sigmoid(xTrain,theta2)
	for i in range(len(pred3Train)):
		if pred3Train[i,0]>0.5:
			pred3Train[i,0]=1
		else:
			pred3Train[i,0]=0
	count=0
	for i in range(len(pred3Train)):
		if yTrain[i]==pred3Train[i]:
			count+=1
	print("Accuracy on training set with L2 regularisation:",(count/len(pred3Train))*100,"%")

	pred3Val=sigmoid(xVal,theta2)
	for i in range(len(pred3Val)):
		if pred3Val[i,0]>0.5:
			pred3Val[i,0]=1
		else:
			pred3Val[i,0]=0
	count=0
	for i in range(len(pred3Val)):
		if yVal[i]==pred3Val[i]:
			count+=1
	print("Accuracy on validation set with L2 regularisation:",(count/len(pred3Val))*100,"%")
	
	pred3Test=sigmoid(xTest,theta2)
	for i in range(len(pred3Test)):
		if pred3Test[i,0]>0.5:
			pred3Test[i,0]=1
		else:
			pred3Test[i,0]=0
	count=0
	for i in range(len(pred3Test)):
		if yTest[i]==pred3Test[i]:
			count+=1
	print("Accuracy on test set with L2 regularisation:",(count/len(pred3Test))*100,"%")
	
	plt.figure("L1 regularisation")
	plt.title("L1 regularisation")
	plt.plot(arange(iterations),rmse1)
	plt.xlabel("Iterations")
	plt.ylabel("RMSE")
	plt.grid()

	plt.figure("L2 regularisation")
	plt.title("L2 regularisation")
	plt.plot(arange(iterations),rmse2)
	plt.xlabel("Iterations")
	plt.ylabel("RMSE")
	plt.grid()

	plt.show()

if __name__=='__main__':
	run()
