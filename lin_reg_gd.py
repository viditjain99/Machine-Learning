from numpy import *
import matplotlib.pyplot as plt
from pandas import *
from math import *
from numpy.linalg import inv

def findRMSE(n,err):
	return ((1/n)*dot(err.transpose(),err))**0.5

def gradientDescent(x,y,theta,iterations,learning_rate):
	rmse=zeros(iterations)
	n=len(x)
	y=y.reshape(y.shape[0],1)
	for i in range(iterations):
		pred=dot(x,theta)
		err=pred-y
		rmse[i]=findRMSE(n,err)
		theta=theta-(learning_rate*(1/n)*dot(x.transpose(),err))
	return theta,rmse

def gradientDescentWithL1(x,y,theta,iterations,learning_rate,alpha):
	rmse=zeros(iterations)
	n=len(x)
	y=y.reshape(y.shape[0],1)
	for i in range(iterations):
		pred=dot(x,theta)
		err=pred-y
		rmse[i]=findRMSE(n,err)
		theta=theta-(learning_rate*((1/n)*dot(x.transpose(),err)))-((learning_rate*alpha*sign(theta))/n)
	return theta,rmse
	
def gradientDescentWithL2(x,y,theta,iterations,learning_rate,alpha):
	rmse=zeros(iterations)
	n=len(x)
	y=y.reshape(y.shape[0],1)
	for i in range(iterations):
		pred=dot(x,theta)
		err=pred-y
		rmse[i]=findRMSE(n,err)
		theta=theta*(1-(alpha/n))-((learning_rate/n)*(dot(x.transpose(),err)))
	return theta,rmse

def run():
	data=read_csv('data.csv')
	y=data.iloc[:,1].values
	y.reshape(y.shape[0],1)
	learning_rate=0.0001
	alpha=5
	iterations=100000
	x=data.iloc[:,0:1]
	xCopy=copy(x)
	a=ones((x.shape[0],1),dtype='double')
	x=append(a,x,axis=1)
	theta=zeros((len(x[0]),1))
	theta,cost=gradientDescent(x,y,theta,iterations,learning_rate)
	print("Gradient descent without regularisation:")
	print("Slope: ",theta[1][0])
	print("Intercept: ",theta[0][0])
	
	y1=zeros((len(xCopy),1))
	count=0
	for i in range(len(xCopy)):
		res=theta[1]*xCopy[i]+theta[0]
		y1[i]=res
	err=y1-y
	print("RMSE: ",findRMSE(len(y1),err)[0][0])
	print()
	plt.figure("Without regularisation")
	plt.title("Without regularisation")
	plt.scatter(xCopy,y)
	plt.plot(xCopy,y1,'r')
	plt.xlabel("Brain Weight")
	plt.ylabel("Body Weight")
	plt.grid()


	#L1
	theta2=zeros((len(x[0]),1))
	theta2,cost2=gradientDescentWithL1(x,y,theta2,iterations,learning_rate,alpha)
	print("Gradient descent with L1 regularisation:")
	print("Slope: ",theta2[1][0])
	print("Intercept: ",theta2[0][0])
	y3=zeros((len(xCopy),1))
	for i in range(len(xCopy)):
		res=theta2[1]*xCopy[i]+theta2[0]
		y3[i]=res
	err=y3-y
	print("RMSE: ",findRMSE(len(y3),err)[0][0])
	print()
	plt.figure("L1 regularisation")
	plt.title("L1 regularisation")
	plt.scatter(xCopy,y)
	plt.plot(xCopy,y3,'r')
	plt.xlabel("Brain Weight")
	plt.ylabel("Body Weight")
	plt.grid()

	#L2
	theta1=zeros((len(x[0]),1))
	theta1,cost1=gradientDescentWithL2(x,y,theta1,iterations,learning_rate,alpha)
	print("Gradient descent with L2 regularisation:")
	print("Slope: ",theta1[1][0])
	print("Intercept: ",theta1[0][0])
	y2=zeros((len(xCopy),1))
	for i in range(len(xCopy)):
		res=theta1[1]*xCopy[i]+theta1[0]
		y2[i]=res
	err=y2-y
	print("RMSE: ",findRMSE(len(y2),err)[0][0])
	print()
	plt.figure("L2 regularisation")
	plt.title("L2 regularisation")
	plt.scatter(xCopy,y)
	plt.plot(xCopy,y2,'r')
	plt.xlabel("Brain Weight")
	plt.ylabel("Body Weight")
	plt.grid()

	plt.show()

if __name__=='__main__':
	run()