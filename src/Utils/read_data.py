'''
simple function to read and pre-process the data
'''

import os
import re
import pandas
import numpy as np
import time
import keras
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
from keras.regularizers import *

SELEX_LEN = 20
PBM_LEN_TOTAL = 60
PBM_LEN = 36

class DataPipeline(object):

	# argsDist {trainPercentage: num, isPadding: bool}
	def __init__(self, dataRoot='', mode='Train', argsDist={}):

		print('+++++++++ DataPipeline was created +++++++++')
		self.dataRoot = dataRoot
		self.mode = mode
		self.argsDict = argsDist
		self.extract_samples_list()
		self.currentSample = self.get_sample_number(mode='debug')
		self.trainData, self.validationData, self.trainLabel, self.validationLabel, self.testData = self.get_data_and_labels_for_sample_number()

	#########################################################################
	# Description: The function reads all the files in a root folder
	# Input: trainDataRoot - path to the folder that contains the files
	# Output: filenamesList, numberOfSamples
	#########################################################################
	def extract_samples_list(self):
		self.filenamesList = os.listdir(self.dataRoot)

		# splitedFilenamesList contains the beginnings of the different files, like TP<num>
		filteredFilenamesList = [filename.split('_')[0][2:] for filename in self.filenamesList]
		sampleNumbersList = list(set(filteredFilenamesList))
		sampleNumbersList = [int(num) for num in sampleNumbersList]

		self.numberOfSamples = len(sampleNumbersList)
		maxSampleNumber = max(sampleNumbersList)
		minSampleNumber = min(sampleNumbersList)

		# TODO:
		assert (self.numberOfSamples == maxSampleNumber-minSampleNumber+1), "Num of samples is not ,matched"

		return

	def get_sample_number(self, mode):
		if mode == 'debug':
			return 1
		elif mode == 'random':
			return int(np.random.randint(min(self.numberOfSamples), max(self.numberOfSamples), 1))
		else:
			raise ValueError('No such a mode {}'.format(str(mode)))

	def get_data_and_labels_for_sample_number(self):
		currTime = time.time()
		# Read PBM test file:
		pbmFile = ''
		testFileName = 'TF{}_pbm.txt'.format(self.currentSample)
		if testFileName not in self.filenamesList:
			print('Error - PBM file not exist for sample number {}.'.format(self.currentSample))
		else:
			pbmFile = self.read_pbm_file(os.path.join(self.dataRoot, testFileName))

		# Read the SELEX files:
		selexFiles = []
		for i in range(10):
			selexFileName = 'TF{}_selex_{}.txt'.format(self.currentSample, i)
			if (i == 0) and (testFileName not in self.filenamesList):
					print('Error - PBM file not exist for sample number {}.'.format(self.currentSample))
			if selexFileName in self.filenamesList:
				selexFiles.append(self.read_selex_file(os.path.join(self.dataRoot, selexFileName)))
			else:
				break
		selexFiles = np.array(selexFiles)
		print('Loaded PBM and SELEX files for sample number: {}.\nNumber of selex files: {}.'.format(self.currentSample, i))
		endTime = time.time()
		print('Loading the data took {} seconds.'.format(round(endTime-currTime, 2)))
		trainData, validationData, trainLabel, validationLabel = self.process_selex_data(selexFiles)
		testData = self.process_PBM_data(pbmFile)

		return trainData, validationData, trainLabel, validationLabel, testData

	#########################################################################
	# Description: Naive implementation for extracting SELEX data.
	#              Cycle '0' will be labeled as negative and the other cycles as positive
	#
	#              Basically the function generates the train and validation data
	# Input: selexsFilesList
	# Output: trainData, validationData, trainLabel, validationLabel
	#########################################################################
	def process_selex_data(self, selexsFilesList):

		# Create united data array and label array
		# Naive assumption - cycle 0 is negative class.
		numberLabelPositive = np.sum(len(selexsFilesList[num]) for num in range(1, len(selexsFilesList)))
		numberLabelNegative = len(selexsFilesList[0])

		labelPositive = np.ones([numberLabelPositive, 1])
		labelNegative = np.zeros([numberLabelNegative, 1])

		# Label of all Selex data
		label = np.concatenate((labelNegative, labelPositive), axis=0)

		selexArray = np.concatenate(selexsFilesList, axis=0) # shape: [num_of_rows, 2]

		# Extract only the strings without the 'count' value
		#  TODO: maybe use the count value ?
		data = selexArray[:, 0].reshape([-1, 1])

		# Shuffle the Selex data
		union = np.concatenate((data, label), axis=1)
		union = np.random.permutation(union)
		data, label = np.split(union, 2, axis=1)


		# Divide into train and validation datasets
		trainPercentage = self.argsDict.pop('trainPercentage', 0.7)
		slice = round(trainPercentage*data.shape[0])
		trainData, validationData = data[:slice,:], data[slice:,:]
		trainLabel, validationLabel = label[:slice], label[slice:]
		print('Train dimensions: {}.\nValidation dimensions: {}.'.format(np.shape(trainData), np.shape(validationData)))

		return trainData, validationData, trainLabel, validationLabel

	# placeholder function
	def process_PBM_data(self, pbmData):
		return map(oneHot, pbmData)

	# Creates a random permutation vector of ind in size of train data
	def init_batching_for_epoch(self):
		self.permutatedInd = np.random.permutation(np.shape(self.trainData)[0])
		self.batchStartInd = 0
		return

	# Extracts a randomly selected batch for train step
	# output two numpy array one for samples and one for labels
	def next_batch_from_epoch(self):
		batchSize = self.argsDict.pop('batchSize', 16)
		# TODO - extract only the strigs without the count column
		batchSamples = self.trainData[self.permutatedInd[self.batchStartInd: self.batchStartInd + batchSize], 0]
		batchLabels = self.trainData[self.permutatedInd[self.batchStartInd: self.batchStartInd + batchSize], 2]
		self.batchStartInd += batchSize

		return batchSamples, batchLabels

	@staticmethod
	def read_pbm_file(pbmFilePath):
		with open(pbmFilePath, 'r') as file:
			text = file.read()
		text = text.split('\n')
		return text[:-1]

	@staticmethod
	def read_selex_file(selexFilePath):
		with open(selexFilePath, 'r') as file:
			text = file.read()
		text = text.split('\n')

		# split to two columns of string and count
		splitedText = [row.split('\t') for row in text]
		return splitedText[:-1]

########################
# HELPER FUNCTIONS 
########################
# def convert_dna_string_to_matrix(dnaString):
# 	matrix = list()
# 	for base in dnaString:
# 		if base == 'A':
# 			base_encoding = [1, 0, 0, 0]
# 		elif base == 'C':
# 			base_encoding = [0, 1, 0, 0]
# 		elif base == 'G':
# 			base_encoding = [0, 0, 1, 0]
# 		elif base == 'T':
# 			base_encoding = [0, 0, 0, 1]
# 		else:
# 			raise ValueError
# 		matrix.append(base_encoding)
# 	yield np.array(matrix).T

'''
Input matrix:             Output matrix: padding of 2 columns each side
								
[[0, 0, 0, 1, 0, 1],     [[0,0, 0, 0, 0, 1, 0, 1,0,0],
 [1, 0, 0, 0, 1, 0],      [0,0, 1, 0, 0, 0, 1, 0,0,0], 
 [0, 0, 1, 0, 0, 0],      [0,0, 0, 0, 1, 0, 0, 0,0,0],
 [0, 1, 0, 0, 0, 0]]      [0,0, 0, 1, 0, 0, 0, 0,0,0]]
'''
# # maxSize - the max number of columns (represents char)
# def vertical_zero_pad_matrix(matrix, maxSize):
# 	H, W = np.shape(matrix)
# 	colNum = (maxSize - W) // 2
# 	assert (colNum == (maxSize - W) / 2), 'Invalid size of one-hot matrix of DNA sequence'
#
# 	paddingMatrix = np.zeros((H, colNum))
# 	return np.concatenate((paddingMatrix, matrix, paddingMatrix), axis=1)

def oneHot(string):
	trantab = str.maketrans('ACGT','0123')
	string = string[0] + 'ACGT'
	data = list(string.translate(trantab))
	return to_categorical(data)[:-4]

def oneHotZeroPad(string, maxSize=PBM_LEN):
	trantab = str.maketrans('ACGT','0123')
	string = string[0] + 'ACGT'
	data = list(string.translate(trantab))

	# Convert to one-hot matrix: Lx4
	matrix = to_categorical(data)[:-4]

	# Zero pad to maxSizex4
	length = len(string) - 4
	pad = (maxSize - length) // 2
	if ((maxSize - length) // 2 != (maxSize - length) / 2):
		raise ValueError ('Cannot zero pad the string')
	padMatrix = np.zeros([pad, 4])
	return np.concatenate((padMatrix, matrix, padMatrix), axis=0)

class DataGenerator(keras.utils.Sequence):
	def __init__(self, data, label, batch_size=32, dim=(36, 4), n_channels=1,n_classes=1, shuffle=True):
		self.dim = dim
		self.batch_size = batch_size
		self.label = label
		self.data = data
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.data) / self.batch_size))

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.data))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

		batchData = [self.data[k] for k in indexes]
		batchData = list(map(oneHotZeroPad, batchData))
		batchData = np.stack(batchData, axis=0)
		batchLabel = np.array([self.label[k] for k in indexes]).reshape(self.batch_size, self.n_classes)
		return batchData, batchLabel


########################
# DEBUG
########################
debugPath = '/Users/royhirsch/Documents/GitHub/BioComp/train_data/'
dataObj = DataPipeline(dataRoot=debugPath, mode='Train', argsDist={})

training_generator = DataGenerator(dataObj.trainData, dataObj.trainLabel)
validation_generator = DataGenerator(dataObj.validationData, dataObj.validationLabel)

model=Sequential()
model.add(Conv1D(filters=4, kernel_size=3, strides=1, kernel_initializer='RandomNormal', activation='relu',
                 input_shape=(36, 4), use_bias=True, bias_initializer='RandomNormal'))
model.add(Flatten())
model.add(Dense(1))

sgd = SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mse')
print('Start training the model.')
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6,
                    verbose=1)
# dnaString = ['ACAAGTTATG']
# mactrixG = oneHotZeroPad(dnaString)

