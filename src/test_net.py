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


def _main(numOfRuns=2):

	# Create logger obj, print with the function : logging.info
	runFolderDir = util_functions.startLogging(isDump=False)

	# Run the nn over num of samples
	resDict = {}
	for sample in range(numOfRuns):

		# Get random sample data and export the files as list
		sampleNum, filesList = util_functions.getTrainSample(dataRoot=os.path.realpath(__file__ + '/../../')+'/train')

		# Create data pipeline obj
		dataPipe = read_data.DataPipeline(filesList)

		# Create and train the model
		# TODO Call your model #

		# Evaluate the model
		AUPR = util_functions.getAUPR(dataPipe.testData, predictions, False)
		resDict[str(sampleNum)] = round(AUPR,5)

	print('########################################\n Results\n########################################')
	printDict(paramsDict)
	for k,v in resDict.items():
		print('Sample number {} : AUPR : {}'.format(k, v))
	print('Average AUPR : {}'.format(sum(resDict.values())/len(resDict)))

if __name__ == '__main__':
	_main()
