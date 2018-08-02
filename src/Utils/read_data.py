import os
import time
import keras
import sys
from keras.utils import to_categorical
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
from Utils.util_functions import get_model_parameters

########################
# CONSTANTS
########################
SELEX_LEN = 20
PBM_LEN_TOTAL = 60
PBM_LEN = 36

########################
# LOAD PARAMS
# Reads external json file with parameters.
########################
params_file_name = os.path.abspath(__file__ + '/../') + '/config.json'
parameters = get_model_parameters(params_file_name)

#########################################################################
# Description: This class manages the data loading of the model.
# 			   It gets the PBM and SELEX path as arguments and generates the data generators.
# 	           The relevant train and test data per experiment.
#
# Input: listOfSysArgs - from the calling to the main.py script
# Output: DataPipeline obj
#########################################################################
class DataPipeline(object):

	def __init__(self, listOfSysArgs):
		#print('+++++++++ DataPipeline was created +++++++++')
		self.TF_index = listOfSysArgs[1].split('_')[0][2:]
		# Load and pre-process the data
		self.trainData, self.validationData, self.trainLabel, self.validationLabel, self.testData = \
			self._get_data_and_labels_for_sample_number(listOfSysArgs)

		# Create generators for the data
		self.train_generator = DataGenerator(self.trainData, self.trainLabel, parameters['batch_size'],
		                                     parameters['input_shape'], True, self.selex_size)

		self.validation_generator = DataGenerator(self.validationData, self.validationLabel, parameters['batch_size'],
		                                     parameters['input_shape'], True, self.selex_size)

		self.test_generator = TestDataGenerator(self.testData, parameters['batch_size'],
		                                     parameters['input_shape'], False, self.selex_size)

		self.input_shape  = (max(PBM_LEN,self.selex_size),4)
	#########################################################################
	# Description: Read and pre-process the SELEX and PBM data.
	# Input: listOfSysArgs
	# Output: trainData, validationData, trainLabel, validationLabel, testData
	#########################################################################
	def _get_data_and_labels_for_sample_number(self, listOfSysArgs):

		# TODO: needs to delete before submition !
		# Get the absolute path of the Selex and PBM files
		trainDataRoot = os.path.realpath(__file__ + "/../../../") + '/train/'

		pbmFilePath = os.path.abspath(os.path.join(trainDataRoot, listOfSysArgs[1]))
		selexFilesPathList = []
		for file in range(2, len(listOfSysArgs)):
			selexFilesPathList.append(os.path.abspath(os.path.join(trainDataRoot, listOfSysArgs[file])))

		# Read PBM file:
		currTime = time.time()
		pbmFile = self.read_pbm_file(pbmFilePath)

		# Read the SELEX files:
		selexFiles = []
		for selexPath in selexFilesPathList:
			selexFiles.append(self.read_selex_file(selexPath))
		#print('Loaded PBM and SELEX files')
		endTime = time.time()
		#print('Loading the data took {} seconds.'.format(round(endTime-currTime, 2)))
		self.selex_size = len(selexFiles[0][0][0])
		# Pre-process the data:
		trainData, validationData, trainLabel, validationLabel = self.process_selex_data(selexFiles)
		testData = self.process_PBM_data(pbmFile)

		return trainData, validationData, trainLabel, validationLabel, testData

	#########################################################################
	# Description: Naive implementation for extracting SELEX data.
	#              Cycle '0' will be labeled as negative and the last cycle as positive.
	#              Basically the function generates the train and validation data.
	#
	# Input: selexsFilesList
	# Output: trainData, validationData, trainLabel, validationLabel
	#########################################################################
	def process_selex_data(self, selexsFilesList):

		# Cycle 0 is False and the last cycle in True
		numberLabelPositive = len(selexsFilesList[-1])
		numberLabelNegative = len(selexsFilesList[0])

		labelPositive = np.ones([numberLabelPositive, 1])
		labelNegative = np.zeros([numberLabelNegative, 1])

		# Filter the data to create equal distribution of the positive and negative labels.
		minNum = min(numberLabelPositive, numberLabelNegative)
		if numberLabelPositive == minNum:
			label = np.concatenate((labelNegative[:minNum,:], labelPositive), axis=0)
			selexArray = np.concatenate([selexsFilesList[0][:minNum], selexsFilesList[-1]], axis=0)

		else:
			label = np.concatenate((labelNegative, labelPositive[:minNum,:]), axis=0)
			selexArray = np.concatenate([selexsFilesList[0], selexsFilesList[-1][:minNum]], axis=0)

		# Extract only the strings without the 'count' value
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

		#print('Train dimensions: {}.\nValidation dimensions: {}.'.format(np.shape(trainData), np.shape(validationData)))

		return trainData, validationData, trainLabel, validationLabel

	# Placeholder function for processing of PBM data
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
#              Generator is a Python data structer that being evaluated only during runtime.
#              It allows to load and pre-process large amount of data in efficient way
#              and parallel to the models training.
#
# Input: data, label, batch_size, dim, compliment, shuffle
# Output: DataGenerator obj
#########################################################################
class DataGenerator(keras.utils.Sequence):
	def __init__(self, data, label, batch_size, dim, shuffle, selex_size):
		self.dim = dim
		self.batch_size = batch_size
		self.label = label
		self.data = data
		self.n_channels = 1
		self.n_classes = 1
		self.shuffle = shuffle
		self.on_epoch_end()
		self.selex_size = selex_size

	def __len__(self):
		# Denotes the number of batches per epoch
		return int(np.floor(len(self.data) / self.batch_size))

	def on_epoch_end(self):
		# Updates indexes after each epoch, to shuffle the training data.
		self.indexes = np.arange(len(self.data))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	#########################################################################
	# Description: Generate one batch of data, evaluates and pre-process the data.
	#              While calling keras function 'fit_generator' this method is being called
	#              for each step to generate the batch data.
	#
	# Input: index - starting index to extract the batch from
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
	def __init__(self, data, batch_size, dim, shuffle, selex_size):
		self.dim = dim
		self.batch_size = batch_size
		self.data = data
		self.n_channels = 1
		self.n_classes = 1
		self.shuffle = shuffle
		self.on_epoch_end()
		self.selex_size = selex_size

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
		batchData = list(map(lambda batch: oneHotZeroPadPBM(batch,self.selex_size),batchData))
		batchData = np.stack(batchData, axis=0)
		return batchData

########################
# HELPER FUNCTIONS
########################

#########################################################################
# Description: Translate the string into a one hot encoded matrix.
#              Cut the end of each string (to keep only 36 chars),
#              if string is smaller then maxSize,, pad the matrix.
# Input: string
# Output: matrix [max(PBM_LEN,selex_size), 4]
#########################################################################
def oneHotZeroPadPBM(string,selex_size):
	maxSize = max(PBM_LEN, selex_size)
	trantab = str.maketrans('ACGT', '0123')
	string = string[:36] + 'ACGT'
	data = list(string.translate(trantab))

	matrix = to_categorical(data)[:-4]

	# Pad to maxSizex4
	length = PBM_LEN
	if maxSize==length:
		return matrix
	else:
		pad = (maxSize - length) // 2
		if ((maxSize - length) // 2 != (maxSize - length) / 2):
			raise ValueError('Cannot zero pad the string')

		padMatrix = np.full((pad, 4), 0.25)
		# Inactive option for zero pad
		# padMatrix = np.zeros([pad, 4])
		return np.concatenate((padMatrix, matrix, padMatrix), axis=0)

#########################################################################
# Description: Translate the string into a one hot encoded matrix.
#              If string is smaller then maxSize,, pad the matrix.
# Input: string
# Output: matrix [max(PBM_LEN,selex_size), 4]
#########################################################################
def oneHotZeroPad(string):
	maxSize = max(PBM_LEN, len(string[0]))
	trantab = str.maketrans('ACGT', '0123')
	string = string[0] + 'ACGT'
	data = list(string.translate(trantab))

	# Convert to one-hot matrix: Lx4
	matrix = to_categorical(data)[:-4]

	# Pad to maxSizex4
	length = len(string) - 4
	if maxSize==length:
		return matrix
	else:
		pad = (maxSize - length) // 2
		if ((maxSize - length) // 2 != (maxSize - length) / 2):
			raise ValueError('Cannot zero pad the string')

		padMatrix = np.full((pad, 4), 0.25)
		# Inactive option for zero pad
		# padMatrix = np.zeros([pad, 4])
		return np.concatenate((padMatrix, matrix, padMatrix), axis=0)
