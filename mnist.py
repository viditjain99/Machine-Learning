from numpy import *
from mlxtend.data import loadlocal_mnist
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
def L1(trainImg,trainLabel,testImg,testLabel):
	models=[]
	print("L1 regularisation")
	print("Training...")
	for i in range(10):
		xTrainLabel=trainLabel.copy()
		for j in range(len(xTrainLabel)):
			if xTrainLabel[j]==i:
				xTrainLabel[j]=1
			else:
				xTrainLabel[j]=0

		logisticRegression=LogisticRegression(penalty='l1',solver='liblinear',max_iter=1000)
		logisticRegression.fit(trainImg,xTrainLabel)
		models.append(logisticRegression)
	print('Training complete')

	for i in range(10):
		xTrainLabel=trainLabel.copy()
		for j in range(trainImg.shape[0]):
			if xTrainLabel[j]==i:
				xTrainLabel[j]=1
			else:
				xTrainLabel[j]=0
		count=0
		pred=models[i].predict(trainImg)
		for j in range(len(pred)):
			if pred[j]==xTrainLabel[j]:
				count+=1
		print('Training set accuracy for class ',i,':',(count/len(pred))*100)

	print()
	for i in range(10):
		xTestLabel=testLabel.copy()
		for j in range(testImg.shape[0]):
			if xTestLabel[j]==i:
				xTestLabel[j]=1
			else:
				xTestLabel[j]=0
		count=0
		pred=models[i].predict(testImg)
		for j in range(len(pred)):
			if pred[j]==xTestLabel[j]:
				count+=1
		print('Testing set accuracy for class ',i,':',(count/len(pred))*100)

def L2(trainImg,trainLabel,testImg,testLabel):
	models=[]
	print("L2 regularisation")

	print("Training...")
	for i in range(10):
		xTrainLabel=trainLabel.copy()
		for j in range(len(xTrainLabel)):
			if xTrainLabel[j]==i:
				xTrainLabel[j]=1
			else:
				xTrainLabel[j]=0

		logisticRegression=LogisticRegression(penalty='l2',solver='liblinear',max_iter=1000)
		logisticRegression.fit(trainImg,xTrainLabel)
		models.append(logisticRegression)
	print('Training complete')

	for i in range(10):
		xTrainLabel=trainLabel.copy()
		for j in range(trainImg.shape[0]):
			if xTrainLabel[j]==i:
				xTrainLabel[j]=1
			else:
				xTrainLabel[j]=0
		count=0
		pred=models[i].predict(trainImg)
		for j in range(len(pred)):
			if pred[j]==xTrainLabel[j]:
				count+=1
		print('Training set accuracy for class ',i,':',(count/len(pred))*100)

	print()	
	falsePos=dict()
	truePos=dict()
	areaUnderCurve=dict()
	acc=zeros(10,1)
	for i in range(10):
		xTestLabel=testLabel.copy()
		for j in range(testImg.shape[0]):
			if xTestLabel[j]==i:
				xTestLabel[j]=1
			else:
				xTestLabel[j]=0
		count=0
		pred=models[i].predict(testImg)
		for j in range(len(pred)):
			if pred[j]==xTestLabel[j]:
				count+=1
		print('Accuracy for class ',i+1,':',(count/len(pred))*100)
		acc[i]=count/len(pred)
		falsePos[i],truePos[i]=roc_curve(xTestLabel,count/len(pred))
		areaUnderCurve[i]=auc(falsePos[i],truePos[i])

		falsePos["micro"],truePos["micro"],_ =roc_curve(xTestLabel.ravel(),acc.ravel())
		areaUnderCurve["micro"]=auc(falsePos["micro"],truePos["micro"])

		allFalsePos=unique(concatenate(falsePos[i] for i in range(10)))
		meanTruePos=zeros_like(allFalsePos)
		for i in range(10):
			meanTruePos+=interp(allFalsePos,falsePos[i],truePos[i])
		meanTruePos/=10
		falsePos["macro"]=allFalsePos
		truePos["macro"]=meanTruePos
		areaUnderCurve["macro"]=auc(falsePos["macro"],truePos["macro"])
		plt.figure()
		plt.plot(falsePos["micro"],truePos["micro"],
	         label='micro-average ROC curve (area = {0:0.2f})'
	               ''.format(areaUnderCurve["micro"]),
	         color='deeppink', linestyle=':', linewidth=4)

		plt.plot(falsePos["macro"],truePos["macro"],
		         label='macro-average ROC curve (area = {0:0.2f})'
		               ''.format(areaUnderCurve["macro"]),
		         color='navy', linestyle=':', linewidth=4)

		colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
		for i, color in zip(range(10), colors):
		    plt.plot(falsePos[i],truePos[i], color=color, lw=lw,
		             label='ROC curve of class {0} (area = {1:0.2f})'
		             ''.format(i, areaUnderCurve[i]))

		plt.plot([0, 1], [0, 1], 'k--', lw=lw)
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Some extension of Receiver operating characteristic to multi-class')
		plt.legend(loc="lower right")
		plt.show()


if __name__=='__main__':
	print("Importing datasets")
	trainImg,trainLabel=loadlocal_mnist('train-images-idx3-ubyte','train-labels-idx1-ubyte')
	testImg,testLabel=loadlocal_mnist('t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte')
	trainImg=(trainImg-trainImg.mean())/trainImg.std()
	testImg=(testImg-testImg.mean())/testImg.std()
	print("Datasets imported")
	# L1(trainImg,trainLabel,testImg,testLabel)
	print()
	L2(trainImg,trainLabel,testImg,testLabel)