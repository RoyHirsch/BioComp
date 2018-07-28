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
numOfRuns = 4
modelNum = 2

def _main(numOfRuns=20):

	ind = 0
	listOfSamples = [57, 59, 86, 80, 41, 54, 9, 108, 73, 30, 62, 96, 5, 50, 74, 24, 99, 101, 68, 31]

	# Create logger obj, print with the function : logging.info
	runFolderDir = util_functions.startLogging(isDump=False)

	# Permutate paramsDict to
	paramsDict = getParamsDict(modelNum)
	paramsDict['depth'] = [80, 80, 96]
	paramsDict['max_pool'] = 10
	paramsDict['dropout'] = 0.5
	paramsDict['hidden_size'] = 128
	paramsDict['lr_decay'] = 1e-06
	paramsDict['optimizer'] = 'adam'

	printDict(paramsDict)
	print('compliment : False')
	print('batch_size : 64')
	print('balanced_data : True')
	print('Compliment - longer num of steps per epoch')

	# Run the nn over num of samples
	resDict = {}
	for sample in range(numOfRuns):

		# Get random sample data and export the files as list
		sampleNum, filesList = util_functions.getTrainSampleFromList(dataRoot=os.path.realpath(__file__ + '/../../')+'/train',
		                                                             listOfSamples=listOfSamples,
		                                                             ind=ind)
		ind += 1

		# Create data pipeline obj
		dataPipe = read_data.DataPipeline(filesList)

		# Create and train the model
		model = SimpleModel(paramsDict, modelNum)
		model.train(tain_generator=dataPipe.train_generator, validation_generator=None,
		            steps_per_epoch=15000, n_epochs=3, n_workers=6)

		# Evaluate the model
		predictions = model.predict(dataPipe.test_generator, 6)
		AUPR = util_functions.getAUPR(dataPipe.testData, predictions)
		resDict[str(sampleNum)] = round(AUPR, 5)

	print('\n########################################\n Hyper Params\n########################################\n')
	print('Model number {}'.format(modelNum))
	yaronAUPR = [0,0.594610698,0.514773825,0.4717869,0.341958113,0.250109638,0.246641534,0.239458146,0.237103258,0.221448961,0.173986496,0.172595686,0.167862911,0.114530832,0.113215591,0.103344651,0.002081527,0.002065405,0.002049701,0.087953909,0.037662088]
	printDict(paramsDict)
	print('\n########################################\n Results\n########################################\n')
	for k,v in resDict.items():
		print('Sample number {} : my AUPR {} : yaron AUPR {}'.format(k, round(v, 4),round(yaronAUPR[listOfSamples.index(int(k)) + 1],4)))
	print('My average AUPR : {}'.format(round(sum(resDict.values())/len(resDict),4)))
	print('Yarons average AUPR : {}'.format(round(sum(yaronAUPR) / len(yaronAUPR),4)))

if __name__ == '__main__':
	_main()
