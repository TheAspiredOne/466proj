
#AveryTan (altan) 1392212 CMPUT466PROJ



import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.externals import joblib


def cross_validate():
	breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
	breast_cancer_X = breast_cancer_dataset.data 
	breast_cancer_Y = breast_cancer_dataset.target



	#avg used for feature scaling
	bc_avg = np.sum(breast_cancer_X, axis = 0)
	bc_avg /= float(len(breast_cancer_X))

	#range used for normalization
	bc_max = np.max(breast_cancer_X, axis = 0)
	bc_min = np.min(breast_cancer_X, axis = 0)
	bc_range = bc_max - bc_min

	#perform feature scaling and normlization
	breast_cancer_X /= bc_range
	breast_cancer_X -=bc_avg


	#straitified split keeping 10% of data for final evaluation of our selected models
	xtrain, xtest, ytrain, ytest = train_test_split(breast_cancer_X, breast_cancer_Y, test_size = 0.1, stratify = breast_cancer_Y ) 


	#leave-one-out-cross-validation used since sample size is significantly limited. >>30 but <<1000
	loo = LeaveOneOut()
	np.save('xtest',xtest)
	np.save('ytest',ytest)
	np.save('xtrain', xtrain)
	np.save('ytrain', ytrain)


	#Perform leave-one-out-cross-validation on SVM, testing for different values of C, kernel, gamma, degree
	Cs = np.logspace(-6,-1,5)
	gammas = np.logspace(-6,-1,5)
	kernels = ['rbf', 'sigmoid']
	parameters = {'C':Cs, 'gamma':gammas, 'kernel':kernels}

	clf_svm = GridSearchCV(estimator = SVC(), param_grid = parameters, scoring = 'recall' ,cv = loo)
	clf_svm.fit(xtrain,ytrain)

	joblib.dump(clf_svm, 'clf_svm.pkl')

	bestparams_svm = {'C': clf_svm.best_estimator_.C, 'gamma':clf_svm.best_estimator_.gamma, 'kernel':clf_svm.best_estimator_.kernel}
	print(bestparams_svm)
	# bestparams, 'C': 1.0000000000000001e-15, 'gamma': 1.0000000000000001e-15, 'kernel': 'rbf'



	#perform leave-one-out-cross-validation on LogisticReg
	losses = ['hinge', 'log', 'modified_huber','squared_loss']
	alphas = np.logspace(-4,-2,5)
	learning_rates = ['optimal']
	parameters = {'loss':losses, 'alpha':alphas, 'learning_rate':learning_rates}

	clf_SGDclf = GridSearchCV(estimator = SGDClassifier(), param_grid = parameters, scoring = 'recall', cv = loo)
	clf_SGDclf.fit(xtrain,ytrain)
	joblib.dump(clf_SGDclf , 'clf_SGDclf.pkl')


	bestparams_SGDclf = {'loss':clf_SGDclf.best_estimator_.loss, 'alpha':clf_SGDclf.best_estimator_.alpha, 'learning_rate':clf_SGDclf.best_estimator_.learning_rate}
	print(bestparams_SGDclf)
	print(clf_SGDclf.best_score_)
	# bestparams, 'loss': 'modified_huber', 'alpha': 0.001, 'learning_rate': 'optimal'
	# bestscore : 0.412109375



	clf_NB = GaussianNB()
	clf_NB.fit(xtrain,ytrain)
	joblib.dump(clf_NB , 'clf_NB.pkl')

	return 0



def plotROC():

	zero = np.zeros(1)
	ytest = np.load('ytest.npy')
	xtest = np.load('xtest.npy')
	xtrain = np.load('xtrain.npy')
	ytrain = np.load('ytrain.npy')



	#plotting ROC for SGDClassifier
	clf_SGDclf = SGDClassifier(loss= 'modified_huber', alpha= 0.001, learning_rate= 'optimal')
	probas_SGD =  clf_SGDclf.fit(xtrain,ytrain).predict_proba(xtest)
	fpr_SGD,tpr_SGD,thesholds_SGD = roc_curve(ytest, probas_SGD[:,1])
	roc_auc_SGD = auc(fpr_SGD,tpr_SGD)


	#plotting ROC for SVM
	clf_svm = SVC(C = 1.0000000000000001e-15, gamma= 1.0000000000000001e-15, kernel= 'rbf', probability = True)
	probas_SVM = clf_svm.fit(xtrain,ytrain).predict_proba(xtest)
	fpr_SVM, tpr_SVM, thresholds_SVM =  roc_curve(ytest, probas_SVM[:,1])
	roc_auc_SVM = auc(fpr_SVM, tpr_SVM)


	#plotting ROC for NaiveBayes
	clf_NB = GaussianNB()
	probas_NB = clf_NB.fit(xtrain,ytrain).predict_proba(xtest)
	fpr_NB, tpr_NB, thresholds_NB = roc_curve(ytest, probas_NB[:,1])
	# fpr_NB = np.concatenate((zero, fpr_NB))
	# tpr_NB = np.concatenate((zero,tpr_NB))
	roc_auc_NB = auc(fpr_NB, tpr_NB)

	print(tpr_NB,tpr_SVM,tpr_SGD)




	plt.figure()
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')


	# plt.plot([0,1], [0,1], linestyle = '--', lw = 2, color = 'r', label = 'Luck', alpha = 0.8)
	plt.plot(fpr_SVM, tpr_SVM, lw = 2, alpha = 0.5, label= "SVM (AUC = %0.2f)" % (roc_auc_SVM))
	plt.plot(fpr_SGD, tpr_SGD, lw = 2, alpha = 0.5, label = "SGD Classifier (AUC = %0.2f)" % (roc_auc_SGD))
	plt.plot(fpr_NB, tpr_NB, lw = 2, alpha = 0.5, label = 'Naive Bayes (AUC = %0.2f)' % (roc_auc_NB))
	plt.legend(loc="lower right")
	plt.show()



def ttest():
	zero = np.zeros(1)
	ytest = np.load('ytest.npy')
	xtest = np.load('xtest.npy')
	xtrain = np.load('xtrain.npy')
	ytrain = np.load('ytrain.npy')

	#plotting ROC for SGDClassifier
	clf_SGDclf = SGDClassifier(loss= 'modified_huber', alpha= 0.001, learning_rate= 'optimal')
	probas_SGD =  clf_SGDclf.fit(xtrain,ytrain).predict_proba(xtest)	
	fpr_SGD,tpr_SGD,thesholds_SGD = roc_curve(ytest, probas_SGD[:,1])
	roc_auc_SGD = auc(fpr_SGD,tpr_SGD)


	#plotting ROC for SVM
	clf_svm = SVC(C = 1.0000000000000001e-15, gamma= 1.0000000000000001e-15, kernel= 'rbf', probability = True)
	probas_SVM = clf_svm.fit(xtrain,ytrain).predict_proba(xtest)	
	fpr_SVM, tpr_SVM, thresholds_SVM =  roc_curve(ytest, probas_SVM[:,1])
	roc_auc_SVM = auc(fpr_SVM, tpr_SVM)


	#plotting ROC for NaiveBayes
	clf_NB = GaussianNB()
	probas_NB = clf_NB.fit(xtrain,ytrain).predict_proba(xtest)
	fpr_NB, tpr_NB, thresholds_NB = roc_curve(ytest, probas_NB[:,1])
	roc_auc_NB = auc(fpr_NB, tpr_NB)




	fpr_NB = fpr_NB[:2]

	#comparing SGD with SVM
	res_t, res_p = ttest_rel(fpr_SGD,fpr_SVM)
	print('p-value for comparison between SGD and SVM: ', res_p)


	#comparing SGD with NB
	res_t, res_p = ttest_rel(fpr_SGD, fpr_NB)
	print('p-value for comparison between SGD and NB: ', res_p)


	#comparing SVM with NB
	res_t,res_p = ttest_rel(fpr_SVM, fpr_NB)
	print('p-value for comparison between SVM and NB: ', res_p)


	'''
	p-value for comparison between SGD and SVM:  nan
	p-value for comparison between SGD and NB:  0.5
	p-value for comparison between SVM and NB:  0.5
	'''



if __name__ == '__main__':
	# cross_validate()
	
	#load our best models of our 3 algorithms being considered.
	clf1 = joblib.load('clf_SGDclf.pkl')
	clf2 = joblib.load('clf_svm.pkl')
	clf3 = joblib.load('clf_NB.pkl')


	ttest()
	plotROC()





