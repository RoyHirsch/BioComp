from model import Model
import numpy as np
import keras
from keras.layers import Flatten
from Utils import read_data
from Utils import util_functions
from model import *
import sys
import os
sys.path.append(os.path.realpath(__file__ + "/../../"))
from Utils import read_data, util_functions
from simple_model import *

######################
# CONSTANTS
######################
numOfRuns = 2
modelNum = 1

def _main(numOfRuns=numOfRuns):

	# Create logger obj, print with the function : logging.info
	runFolderDir = util_functions.startLogging(isDump=False)

	# Permutate paramsDict to
	paramsDict = getParamsDict(modelNum)
	# TODO !!!!!!!!!!!!
	aramsDict = {'depth': [80,80,96], 'dropout': 0.25, 'hidden_size': 32, 'lr_decay': 0}

	# Run the nn over num of samples
	resDict = {}
	for sample in range(numOfRuns):

		# Get random sample data and export the files as list
		sampleNum, filesList = util_functions.getTrainSample(dataRoot=os.path.realpath(__file__ + '/../../')+'/train')

		# Debug print
		print('Sample number is {}'.format(sampleNum))

		# Create data pipeline obj
		dataPipe = read_data.DataPipeline(filesList)

		# Create and train the model
		model = SimpleModel(paramsDict, modelNum)
		model.train(tain_generator=dataPipe.train_generator, validation_generator=dataPipe.validation_generator,
		            steps_per_epoch=5000, n_epochs=3, n_workers=6)

		# Evaluate the model
		predictions = model.predict(dataPipe.test_generator, 6)
		AUPR = util_functions.getAUPR(dataPipe.testData, predictions)
		resDict[str(sampleNum)] = round(AUPR, 5)

	print('\n########################################\n Hyper Params\n########################################\n')
	print('Model number {}'.format(modelNum))
	printDict(paramsDict)
	print('\n########################################\n Results\n########################################\n')
	for k,v in resDict.items():
		print('Sample number {} : AUPR : {}'.format(k, v))
	print('Average AUPR : {}'.format(sum(resDict.values())/len(resDict)))

if __name__ == '__main__':
	_main()
