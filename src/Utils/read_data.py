'''
simple function to read and pre-process the data
'''

import os
import re
import pandas
import numpy as np

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
		self.trainData, self.valData, self.GT_PBM_file = self.get_data_and_labels_for_sample_number()


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

		tainData, valData = self.process_selex_data(selexFiles)
		GT_PBM_file = self.process_PBM_data(pbmFile)

		return tainData, valData, GT_PBM_file

	#########################################################################
	# Description: Naive implementation for extracting SELEX data.
	#              Cycle '0' will be labeled as negative and the other cycles as positive
	#
	#              Basically the function generates the train and validation data
	# Input: selexsFilesList
	# Output: filenamesList, numberOfSamples
	#########################################################################
	def process_selex_data(self, selexsFilesList):

		labelCycleOne = np.zeros([len(selexsFilesList[0]), 1])
		numOfSelexCyclesFromOne = np.sum(len(selexsFilesList[num]) for num in range(1, len(selexsFilesList)))
		selexLabel = np.concatenate([labelCycleOne, np.ones([numOfSelexCyclesFromOne, 1])], axis=0)

		selexArray = np.concatenate(selexsFilesList, axis=0) # shape: [num_of_rows, 2]
		# selexAndLabelArray = np.concatenate([selexArray, selexLabel], axis=1) # shape: [num_of_rows, 3]
		# selexAndLabelArray = np.random.permutation(selexAndLabelArray)

		# TODO: extract only the strings without the 'count' value
		selexData = selexArray[:, 0]

		# Encode into one-hot matrix. From array in shape [N, 1] to [N, ,30 ,4]
		isPadding = self.argsDict.pop('isPadding', True)
		selexEncodedData = []
		print('Starting to encode SELEX data into one-hot matrix.')
		for string in selexData:
			matrix = convert_dna_string_to_matrix(string)
			selexEncodedData.append(matrix)
		selexEncodedData = np.concatenate(selexEncodedData, axis=0)
		if isPadding:
			selexEncodedDataPaded = vertical_zero_pad_matrix(selexEncodedData, PBM_LEN)

		print('Finished encode SELEX data into one-hot matrix.')
		print('SELEX data shape is {}.'.format(np.shape(selexEncodedData)))

		trainPercentage = self.argsDict.pop('trainPercentage', 0.7)
		numSamples = np.sum(len(selexsFilesList[num]) for num in range(len(selexsFilesList)))

		# TODO: currently not using 'count' column
		if trainPercentage:
			trainData = selexEncodedData[:int(trainPercentage * numSamples), :]
			valData = selexEncodedData[int(trainPercentage * numSamples):, :]

			trainLabel = selexLabel[:int(trainPercentage * numSamples), :]
			valLabel = selexLabel[int(trainPercentage * numSamples):, :]

		else:
			trainData = selexEncodedData
			valData = []

		print('Train dimensions: {}.\nVal dimensions: {}.'.format(np.shape(trainData), np.shape(valData)))
		return trainData, valData

	# placeholder function
	def process_PBM_data(self, pbmData):
		pbmEncodedData = []
		print('Starting to encode PBM data into one-hot matrix.')
		for string in pbmData:
			matrix = convert_dna_string_to_matrix(string)

			pbmEncodedData.append(matrix)
		pbmEncodedData = np.concatenate(pbmEncodedData, axis=0)
		print('Finished encode PBM data into one-hot matrix.')
		print('PBM data shape is {}.'.format(np.shape(pbmEncodedData)))

		return pbmEncodedData

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
def convert_dna_string_to_matrix(dnaString):
	matrix = list()
	for base in dnaString:
		if base == 'A':
			base_encoding = [1, 0, 0, 0]
		elif base == 'C':
			base_encoding = [0, 1, 0, 0]
		elif base == 'G':
			base_encoding = [0, 0, 1, 0]
		elif base == 'T':
			base_encoding = [0, 0, 0, 1]
		else:
			raise ValueError
		matrix.append(base_encoding)
	return np.array(matrix).T

'''
Input matrix:             Output matrix: padding of 2 columns each side
								
[[0, 0, 0, 1, 0, 1],     [[0,0, 0, 0, 0, 1, 0, 1,0,0],
 [1, 0, 0, 0, 1, 0],      [0,0, 1, 0, 0, 0, 1, 0,0,0], 
 [0, 0, 1, 0, 0, 0],      [0,0, 0, 0, 1, 0, 0, 0,0,0],
 [0, 1, 0, 0, 0, 0]]      [0,0, 0, 1, 0, 0, 0, 0,0,0]]
'''
# maxSize - the max number of columns (represents char)
def vertical_zero_pad_matrix(matrix, maxSize):
	H, W = np.shape(matrix)
	colNum = (maxSize - W) // 2
	assert (colNum == (maxSize - W) / 2), 'Invalid size of one-hot matrix of DNA sequence'

	paddingMatrix = np.zeros((H, colNum))
	return np.concatenate((paddingMatrix, matrix, paddingMatrix), axis=1)


	pass

########################
# DEBUG
########################
debugPath = '/Users/royhirsch/Documents/GitHub/BioComp/train_data/'
dataObj = DataPipeline(dataRoot=debugPath, mode='Train', argsDist={})
dataObj.init_batching_for_epoch()
batchSamples, batchLabels = dataObj.next_batch_from_epoch()
# debugPath2 = '/Users/royhirsch/Documents/GitHub/BioComp/train_data/TF1_selex_0.txt'
# text = DataPipeline.read_selex_file(debugPath2)
