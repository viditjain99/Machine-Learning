from numpy import *
import matplotlib.pyplot as plt
from pandas import *
from math import *
from numpy.linalg import inv

def findRMSE(n,err):
	return ((1/n)*dot(err.transpose(),err))**0.5

def gradientDescent(x,y,theta,iterations,learning_rate,xTest,yTest):
	rmse=[]
	rmseTest=[]
	n=len(x)
	y=y.reshape(y.shape[0],1)
	yTest=yTest.reshape(yTest.shape[0],1)
	for i in range(iterations):
		pred=dot(x,theta)
		err=pred-y
		rmse.append(findRMSE(n,err))
		predTest=dot(xTest,theta)
		errTest=predTest-yTest
		rmseTest.append(findRMSE(n,errTest))
		theta=theta-(learning_rate*(2/n)*dot(x.transpose(),err))
	return theta,rmse,rmseTest

def run():
	data=read_csv('Dataset.csv',names=["sex","length","diameter","height","whole_height","shucked_weight","viscera_weight","shell_weight","rings"])
	learning_rate=0.01
	iterations=5000
	y=data.iloc[:,8].values
	data=(data-data.mean())/data.std()
	data.head()
	x=data.iloc[:,0:8]
	a=ones((x.shape[0],1),dtype='int')
	x=append(a,x,axis=1)
	testStart=0
	testEnd=835
	thetaAns=[]
	trainRmseList=[]
	testRmseList=[]
	for i in range(5):
		theta=zeros((len(x[0]),1))
		xTrain=empty([2,2])
		xTest=empty([2,2])
		yTrain=empty([2,2])
		yTest=empty([2,2])
		if i==0:
			xTest=x[testStart:testEnd]
			yTest=y[testStart:testEnd]
			xTrain=x[testEnd:]
			yTrain=y[testEnd:]
		elif i==4:
			xTest=x[testStart:len(x)]
			yTest=y[testStart:len(y)]
			xTrain=x[0:testStart]
			yTrain=y[0:testStart]
		else:
			xTest=x[testStart:testEnd]
			yTest=y[testStart:testEnd]
			xTrain=empty(shape(x[1]))
			yTrain=empty(shape(y[1]))
			for j in range(len(x)):
				if not (j>=testStart and j<testEnd):
					xTrain=vstack([xTrain,x[j]])
					yTrain=append(yTrain,y[j])
		testStart=testEnd
		testEnd+=835

		theta,rmseTrain,rmseTest1=gradientDescent(xTrain,yTrain,theta,iterations,learning_rate,xTest,yTest)
		trainRmseList.append(rmseTrain)
		testRmseList.append(rmseTest1)
		thetaAns.append(theta)
		pred=dot(xTest,theta)
		yTest=yTest.reshape(yTest.shape[0],1)
		err=yTest-pred
		rmseTest=findRMSE(len(err),err)

	meanRmse=[]
	for i in range(iterations):
		sum=0
		for j in range(5):
			sum=sum+trainRmseList[j][i][0][0]
		meanRmse.append(sum/5)
	plt.figure("Training set")
	plt.title("Plot for training set")
	plt.xlabel("Iterations")
	plt.ylabel("Mean RMSE")
	plt.grid()
	plt.plot(arange(iterations),meanRmse)

	meanRmse=[]
	for i in range(iterations):
		sum=0
		for j in range(5):
			sum=sum+testRmseList[j][i][0][0]
		meanRmse.append(sum/5)
	plt.figure("Testing set")
	plt.title("Plot for testing set")
	plt.xlabel("Iterations")
	plt.ylabel("Mean RMSE")
	plt.grid()
	plt.plot(arange(iterations),meanRmse)

	finalTheta=[0,0,0,0,0,0,0,0,0]

	for i in range(9):
		finalTheta[i]=thetaAns[0][i][0]+thetaAns[1][i][0]+thetaAns[2][i][0]+thetaAns[3][i][0]+thetaAns[4][i][0]
		finalTheta[i]=finalTheta[i]/5

	y=y.reshape(y.shape[0],1)
	params=dot(inv(dot(x.transpose(),x)),dot(x.transpose(),y))
	rmseTrain=[]
	rmseTest=[]
	testStart=0
	testEnd=835
	for i in range(5):
		xTrain=empty([2,2])
		xTest=empty([2,2])
		yTrain=empty([2,2])
		yTest=empty([2,2])
		if i==0:
			xTest=x[testStart:testEnd]
			yTest=y[testStart:testEnd]
			xTrain=x[testEnd:]
			yTrain=y[testEnd:]
		elif i==4:
			xTest=x[testStart:len(x)]
			yTest=y[testStart:len(y)]
			xTrain=x[0:testStart]
			yTrain=y[0:testStart]
		else:
			xTest=x[testStart:testEnd]
			yTest=y[testStart:testEnd]
			xTrain=empty(shape(x[1]))
			yTrain=empty(shape(y[1]))
			for j in range(len(x)):
				if not (j>=testStart and j<testEnd):
					xTrain=vstack([xTrain,x[j]])
					yTrain=append(yTrain,y[j])
		testStart=testEnd
		testEnd+=835
		yTrain=yTrain.reshape(yTrain.shape[0],1)
		yTest=yTest.reshape(yTest.shape[0],1)

		predTrain=dot(xTrain,params)
		errTrain=yTrain-predTrain
		rmseTrain.append(findRMSE(len(errTrain),errTrain))

		predTest=dot(xTest,params)
		errTest=yTest-predTest
		rmseTest.append(findRMSE(len(errTest),errTest))

	print("RMSE from closed form equation for training set:")
	for i in range(len(rmseTrain)):
		print("For fold ",i+1,": ",rmseTrain[i][0][0])
	print()

	print("RMSE from closed form equation for testing set:")
	for i in range(len(rmseTest)):
		print("For fold ",i+1,": ",rmseTest[i][0][0])

	theta=empty([9,1])
	for i in range(len(finalTheta)):
		theta[i]=finalTheta[i]

	print()
	pred=dot(x,theta)
	err=pred-y
	print("RMSE from gradient descent after convergence: ",findRMSE(len(x),err)[0][0])
	pred=dot(x,params)
	err=pred-y
	print("RMSE from closed form equation: ",findRMSE(len(x),err)[0][0])

	plt.show()
	
if __name__=='__main__':
	run()