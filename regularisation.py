from numpy import *
from pandas import *
from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def findRMSE(n,err):
	return ((1/n)*dot(err.transpose(),err))**0.5

def gradientDescentWithL1(x,y,theta,iterations,learning_rate,alpha):
	rmse=zeros(iterations)
	n=len(x)
	for i in range(iterations):
		pred=dot(x,theta)
		err=pred-y
		rmse[i]=findRMSE(n,err)
		theta=theta-(learning_rate*((1/n)*dot(x.transpose(),err)))-((learning_rate*alpha*sign(theta))/n)
	return theta,rmse
	
def gradientDescentWithL2(x,y,theta,iterations,learning_rate,alpha):
	rmse=zeros(iterations)
	n=len(x)
	for i in range(iterations):
		pred=dot(x,theta)
		err=pred-y
		rmse[i]=findRMSE(n,err)
		theta=theta*(1-(alpha/n))-((learning_rate/n)*(dot(x.transpose(),err)))
	return theta,rmse

def ridgeRegression(x,y,xTest,yTest,alpha):
	params={'alpha':alpha}
	ridge=Ridge()
	gridSearchCV=GridSearchCV(ridge,params,cv=5)
	gridSearchCV.fit(x,y)
	yPred=gridSearchCV.predict(xTest)
	return gridSearchCV.best_params_

def lassoRegression(x,y,xTest,yTest,alpha):
	params={'alpha':alpha}
	lasso=Lasso()
	gridSearchCV=GridSearchCV(lasso,params,cv=5)
	gridSearchCV.fit(x,y)
	yPred=gridSearchCV.predict(xTest)
	return gridSearchCV.best_params_

def run():
	data=read_csv('Dataset.csv',names=["sex","length","diameter","height","whole_height","shucked_weight","viscera_weight","shell_weight","rings"])
	data=(data-data.mean())/data.std()
	x1=data.iloc[:,0:8]
	y1=data.iloc[:,8]

	xTest=x1.iloc[835:1670]
	yTest=y1.iloc[835:1670].values

	temp1X=x1.iloc[0:835]
	temp2X=x1.iloc[1670:]
	temp1Y=y1.iloc[0:835]
	temp2Y=y1.iloc[1670:]

	x=concat([temp1X,temp2X])
	y=concat([temp1Y,temp2Y]).values
	y=y.reshape(y.shape[0],1)
	yTest=yTest.reshape(yTest.shape[0],1)

	alpha=[1e-4,1e-3,1e-2,1e-1,0,5,10,15]
	l1Alpha=lassoRegression(x,y,xTest,yTest,alpha)['alpha']
	l2Alpha=ridgeRegression(x,y,xTest,yTest,alpha)['alpha']
	print('Optimal parameter for lasso regression:',l1Alpha)
	print('Optimal parameter for ridge regression:',l2Alpha)

	a=ones((x.shape[0],1),dtype='double')
	x=append(a,x,axis=1)
	a=ones((xTest.shape[0],1),dtype='double')
	xTest=append(a,xTest,axis=1)
	theta=zeros((len(x[0]),1))
	learning_rate=0.01
	iterations=10000
	theta,cost=gradientDescentWithL1(x,y,theta,iterations,learning_rate,l1Alpha)
	plt.figure("Gradient Descent with L1")
	plt.title("Gradient Descent with L1")
	plt.xlabel('Iterations')
	plt.ylabel('RMSE')
	plt.plot(arange(iterations),cost)
	plt.grid()
	pred=dot(xTest,theta)
	err=yTest-pred
	print("RMSE for test set for L1 regularisation: ",findRMSE(len(xTest),err)[0][0])

	theta=zeros((len(x[0]),1))
	theta,cost=gradientDescentWithL2(x,y,theta,iterations,learning_rate,l2Alpha)
	plt.figure("Gradient Descent with L2")
	plt.title("Gradient Descent with L2")
	plt.xlabel('Iterations')
	plt.ylabel('RMSE')
	plt.plot(arange(iterations),cost)
	plt.grid()
	pred=dot(xTest,theta)
	err=yTest-pred
	print("RMSE for test set for L2 regularisation: ",findRMSE(len(xTest),err)[0][0])

	plt.show()

	

if __name__=='__main__':
	run()