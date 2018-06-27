'''
simple function to read and pre-process the data
'''

import os
import time
import keras
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD

########################
# CONSTANTS
########################
SELEX_LEN = 20
PBM_LEN_TOTAL = 60
PBM_LEN = 36


#########################################################################
# Description: This class manages the data loading to the model.
# 			   If scans dataRoot for all the different files and will generate
# 	           the relevant train and test data per experiment.
# Input: dataRoot - root folder that holds all the files.
#        mode - debug for loading only experiment #1 , operation - real-world operation.
#        argsDist (optional)
#                trainPercentage - num in [0,1]
#                isPadding       - bool
# Output: DataPipeline obj
#########################################################################
class DataPipeline(object):

	def __init__(self, dataRoot='', mode='debug', argsDist={}):

		print('+++++++++ DataPipeline was created +++++++++')
		self.dataRoot = dataRoot
		self.mode = mode
		self.argsDict = argsDist
		self.extract_samples_list()
		self.currentSample = self.get_sample_number()
		self.trainData, self.validationData, self.trainLabel, self.validationLabel,\
		self.testData = self.get_data_and_labels_for_sample_number()

	#########################################################################
	# Description: The function reads all the files in a root folder.
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

		assert (self.numberOfSamples == maxSampleNumber-minSampleNumber+1), "Num of samples is not ,matched"

		return

	#########################################################################
	# Description: Gets the current experiment sample to work on.
	# Input:
	# Output: experiment number
	#########################################################################
	def get_sample_number(self):
		if self.mode == 'debug':
			return 1

		elif self.mode == 'random':
			return int(np.random.randint(min(self.numberOfSamples), max(self.numberOfSamples), 1))

		elif self.mode == 'operation':
			# TODO : finish operaion mode
			pass

		else:
			raise ValueError('No such a mode {}'.format(str(mode)))

	#########################################################################
	# Description: Given an experiment number the function reads the SELEX and PBM data.
	#              It extracts the data and pre-process it if necessary.
	# Input:
	# Output: trainData, validationData, trainLabel, validationLabel, testData
	#########################################################################
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
	def __init__(self, data, label, batch_size=32, dim=(36, 4), n_channels=1, n_classes=1, shuffle=True):
		self.dim = dim
		self.batch_size = batch_size
		self.label = label
		self.data = data
		self.n_channels = n_channels
		self.n_classes = n_classes
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
	def __init__(self, data, batch_size=256, dim=(36, 4), n_channels=1, n_classes=1, shuffle=True):
		self.dim = dim
		self.batch_size = batch_size
		self.data = data
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		# Denotes the number of batches per epoch
		return int(np.floor(len(self.data) / self.batch_size))

	#########################################################################
	# Description: Generate one batch of data, evaluates and pre-process the data.
	# Input:
	# Output: DataGenerator obj
	#########################################################################
	def __getitem__(self, index):
		indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

		batchData = [self.data[k] for k in indexes]
		batchData = list(map(oneHot, batchData))
		batchData = np.stack(batchData, axis=0)
		return batchData

########################
# HELPER FUNCTIONS
########################
def oneHot(string):
	trantab = str.maketrans('ACGT', '0123')
	string = string[0] + 'ACGT'
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
	padMatrix = np.zeros([pad, 4])
	return np.concatenate((padMatrix, matrix, padMatrix), axis=0)

########################
# DEBUG
# Simple model to illustrate the dataPipe behavior
########################
debugPath = '/Users/royhirsch/Documents/GitHub/BioComp/train_data/'
dataObj = DataPipeline(dataRoot=debugPath, mode='Train', argsDist={})

training_generator = DataGenerator(dataObj.trainData, dataObj.trainLabel)
validation_generator = DataGenerator(dataObj.validationData, dataObj.validationLabel)
test_generator = TestDataGenerator(dataObj.testData)

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

# returns np array of predictions
predictions = model.predict_generator(generator=test_generator)