'''
simple function to read and pre-process the data
'''

import os
import time
import keras
import sys
from keras.utils import to_categorical
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
from Utils.parameters import parameters

########################
# CONSTANTS
########################
SELEX_LEN = 20
PBM_LEN_TOTAL = 60
PBM_LEN = 36

########################
# LOAD PARAMS
########################
params_file_name = os.path.abspath(__file__ + '/../') + '/config.json'
parameters = parameters(params_file_name)

#########################################################################
# Description: This class manages the data loading to the model.
# 			   It gets the PBM and SELEX path as arguments and generates the data generators.
# 	           the relevant train and test data per experiment.
# Input: listOfSysArgs
#        argsDist (optional)
#                trainPercentage - num in [0,1]
#                batch_size      - int
#                dim             - tupal of ints
#                n_channels      - int
#                n_classes       - int
# Output: DataPipeline obj
#########################################################################
class DataPipeline(object):

	def __init__(self, listOfSysArgs, selex):
		print('+++++++++ DataPipeline was created +++++++++')
		self.selex = selex

		# Load and pre-process the data
		self.trainData, self.validationData, self.trainLabel, self.validationLabel, self.testData = \
			self._get_data_and_labels_for_sample_number(listOfSysArgs)

		# Create generators for the data
		self.train_generator = DataGenerator(self.trainData, self.trainLabel, parameters['batch_size'],
		                                     parameters['input_shape'], True)

		self.validation_generator = DataGenerator(self.validationData, self.validationLabel, parameters['batch_size'],
		                                     parameters['input_shape'], True)

		self.test_generator = TestDataGenerator(self.testData, parameters['batch_size'],
		                                     parameters['input_shape'], False)


	#########################################################################
	# Description: Read and pre-process the SELEX and PBM data.
	# Input:
	# Output: trainData, validationData, trainLabel, validationLabel, testData
	#########################################################################
	def _get_data_and_labels_for_sample_number(self, listOfSysArgs):

		# TODO: needs to delete before submition !
		# Get the absulote path of the Selex and PBM files
		trainDataRoot = os.path.realpath(__file__ + "/../../../") + '/train/'

		pbmFilePath = os.path.abspath(os.path.join(trainDataRoot, listOfSysArgs[1]))
		selexFilesPathList = []
		for file in range(2, len(listOfSysArgs)):
			selexFilesPathList.append(os.path.abspath(os.path.join(trainDataRoot, listOfSysArgs[file])))

		# Read PBM test file:
		currTime = time.time()
		pbmFile = self.read_pbm_file(pbmFilePath)

		# Read the SELEX files:
		selexFiles = []
		for selexPath in selexFilesPathList:
			selexFiles.append(self.read_selex_file(selexPath))
		print('Loaded PBM and SELEX files')
		endTime = time.time()
		print('Loading the data took {} seconds.'.format(round(endTime-currTime, 2)))

		# Pre-process the data:
		trainData, validationData, trainLabel, validationLabel = self.process_selex_data(selexFiles)
		testData = self.process_PBM_data(pbmFile)

		return trainData, validationData, trainLabel, validationLabel, testData

	#########################################################################
	# Description: Naive implementation for extracting SELEX data.
	#              Cycle '0' will be labeled as negative and the other cycles as positive.
	#              Basically the function generates the train and validation data.
	# Input: selexsFilesList
	# Output: trainData, validationData, trainLabel, validationLabel
	#########################################################################
	def process_selex_data(self, selexsFilesList):

		# Create united data array and label array
		# Naive assumption - cycle 0 is negative class.

		# TODO: ROY 0107 a tryout cycle 0 is False and the last cycle in True
		# numberLabelPositive = np.sum(len(selexsFilesList[num]) for num in range(1, len(selexsFilesList)))
		numberLabelPositive = len(selexsFilesList[self.selex])
		numberLabelNegative = len(selexsFilesList[0])

		labelPositive = np.ones([numberLabelPositive, 1])
		labelNegative = np.zeros([numberLabelNegative, 1])

		# TODO :  A tryout - balanced data


		#minNum = min(numberLabelPositive, numberLabelNegative)
		#if numberLabelPositive == minNum:
		#	label = np.concatenate((labelNegative[:minNum,:], labelPositive), axis=0)
		#	selexArray = np.concatenate([selexsFilesList[0][:minNum], selexsFilesList[self.selex]], axis=0)

		#else:
		#	label = np.concatenate((labelNegative, labelPositive[:minNum,:]), axis=0)
		#	selexArray = np.concatenate([selexsFilesList[0], selexsFilesList[self.selex][:minNum]], axis=0)

		# Label of all Selex data
		label = np.concatenate((labelNegative, labelPositive), axis=0)

		#selexArray = np.concatenate(selexsFilesList, axis=0) # shape: [num_of_rows, 2]
		selexArray = np.concatenate([selexsFilesList[0], selexsFilesList[self.selex]], axis=0)

		# Extract only the strings without the 'count' value
		#  TODO: maybe use the count value ?
		data = selexArray[:, 0].reshape([-1, 1])

		# Shuffle the Selex data
		union = np.concatenate((data, label), axis=1)
		union = np.random.permutation(union)
		data, label = np.split(union, 2, axis=1)

		# Divide into train and validation datasets
		trainPercentage = parameters['train_percentage']
		slice = round(trainPercentage * data.shape[0])
		trainData, validationData = data[:slice,:], data[slice:,:]
		trainLabel, validationLabel = label[:slice], label[slice:]
		print('Train dimensions: {}.\nValidation dimensions: {}.'.format(np.shape(trainData), np.shape(validationData)))

		return trainData, validationData, trainLabel, validationLabel

	# placeholder function
	def process_PBM_data(self, pbmData):
		return pbmData

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

#########################################################################
# Description: Creates 'Generator' object that holds the date pipe for train/val data.
# Input:
# Output: DataGenerator obj
#########################################################################
class DataGenerator(keras.utils.Sequence):
	def __init__(self, data, label, batch_size, dim, shuffle=True):
		self.dim = dim
		self.batch_size = batch_size
		self.label = label
		self.data = data
		self.n_channels = 1
		self.n_classes = 1
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		# Denotes the number of batches per epoch
		return int(np.floor(len(self.data) / self.batch_size))

	def on_epoch_end(self):
		# Updates indexes after each epoch
		self.indexes = np.arange(len(self.data))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	#########################################################################
	# Description: Generate one batch of data, evalutes and pre-process the data.
	# Input:
	# Output: DataGenerator obj
	#########################################################################
	def __getitem__(self, index):
		indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
		batchData = [self.data[k] for k in indexes]
		batchData = list(map(oneHotZeroPad, batchData))
		batchData = np.stack(batchData, axis=0)
		batchLabel = np.array([self.label[k] for k in indexes]).reshape(self.batch_size, self.n_classes)
		return batchData, batchLabel

#########################################################################
# Description: Creates 'Generator' object that holds the date pipe for test data (PBM file).
# Input:
# Output: TestDataGenerator obj
#########################################################################
class TestDataGenerator(keras.utils.Sequence):
	def __init__(self, data, batch_size, dim, shuffle=False):
		self.dim = dim
		self.batch_size = batch_size
		self.data = data
		self.n_channels = 1
		self.n_classes = 1
		self.shuffle = shuffle
		self.on_epoch_end()

	def on_epoch_end(self):
		# Updates indexes after each epoch
		self.indexes = np.arange(len(list(self.data)))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __len__(self):
		# Denotes the number of batches per epoch
		return int(np.ceil(len(list(self.data)) / self.batch_size))

	def __getitem__(self, index):
		indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
		batchData = [self.data[k] for k in indexes]
		batchData = list(map(oneHotPBM, batchData))
		batchData = np.stack(batchData, axis=0)
		return batchData

########################
# HELPER FUNCTIONS
########################
def oneHotPBM(string):
	# Cut the end of each string (to keep only 36 chars)
	trantab = str.maketrans('ACGT', '0123')
	string = string[:36] + 'ACGT'
	data = list(string.translate(trantab))
	return to_categorical(data)[:-4]

def oneHotZeroPad(string, maxSize=PBM_LEN):
	trantab = str.maketrans('ACGT', '0123')
	string = string[0] + 'ACGT'
	data = list(string.translate(trantab))

	# Convert to one-hot matrix: Lx4
	matrix = to_categorical(data)[:-4]

	# Zero pad to maxSizex4
	length = len(string) - 4
	pad = (maxSize - length) // 2
	if ((maxSize - length) // 2 != (maxSize - length) / 2):
		raise ValueError('Cannot zero pad the string')

	padMatrix = np.full((pad, 4), 0.25)
	# padMatrix = np.zeros([pad, 4])

	return np.concatenate((padMatrix, matrix, padMatrix), axis=0)
