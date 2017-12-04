

# Avery Tan (altan) 1392212

import sys
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import datasets 
from sklearn import svm



def normalize_N_scale(pos,neg, master):

	num_features = 30 # we know there are 30 features


	features = master[:,2:] #remove id and label
	features = features.astype(np.float) #convert to floats


	max_f = np.amax(features,axis = 0) 
	min_f = np.amin(features,axis = 0)
	range_f = max_f - min_f

	avg_f = np.sum(features, axis = 0)
	avg_f /= float(master.shape[0])


	for i in pos:
		for j in range(num_features):
			i[j+2] = (float(i[j+2])-avg_f[j])/range_f[j]

	for i in neg:
		for j in range(num_features):
			i[j+2] = (float(i[j+2])-avg_f[j])/range_f[j]

	return (pos,neg)



def process_data(seed = 123):
	'''
	do any pre-processing of data including sorting the data
	'''
	dataset_pos = list()
	dataset_neg = list()
	master_dataset = list()


	f = open("wdbc.data", "r")
	for line in f:
		sample = line.split(',')
		sample[-1] = sample[-1][:-1]


		master_dataset.append(sample)
		if sample[1] == 'M': #sort dataset according to label
			sample[1] = 1.0
			dataset_pos.append(sample)
		else:
			sample[1] = 0.0
			dataset_neg.append(sample)


	np_data_pos = np.asarray(dataset_pos)
	np_data_neg  = np.asarray(dataset_neg)
	np_master = np.asarray(master_dataset)


	#psuedo-random shuffling
	np.random.seed(seed)
	np.random.shuffle(np_data_pos)
	np.random.shuffle(np_data_neg)



	processed_pos, processed_neg = normalize_N_scale(np_data_pos,np_data_neg, np_master)


	return (processed_pos,processed_neg)



def split_labels(pos,neg):
	'''
	keep 10% of the data to serve as the final test dataset
	'''
	pos_indices = int(np.floor(0.1*pos.shape[0]))
	neg_indices = int(np.floor(0.1*neg.shape[0]))

	testSet = np.concatenate((pos[:pos_indices,:], neg[:neg_indices,:]))
	trainSet = np.concatenate((pos[pos_indices:,:], neg[neg_indices:,:]))

	np.random.shuffle(testSet)
	np.random.shuffle(trainSet)


	test_labels = testSet[:,1].astype(np.float)
	train_labels = trainSet[:,1].astype(np.float)

	testSet = testSet[:,2:].astype(np.float)
	trainSet = trainSet[:,2:].astype(np.float)

	return ((trainSet,train_labels),(testSet,test_labels))




if __name__ == '__main__':
	pos,neg = process_data()
	trainSet, testSet = split_labels(pos,neg)

	






