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
#from simple_model import *
from keras.utils import plot_model


def _main(numOfRuns=3):

	# Create logger obj, print with the function : logging.info
	runFolderDir = util_functions.startLogging(isDump=False)

	# Run the nn over num of samples
	resDict = {}
	for sample in range(numOfRuns):

		# Get random sample data and export the files as list
		sampleNum, filesList = util_functions.getTrainSample(dataRoot=os.path.realpath(__file__ + '/../../')+'/train')

		# Create data pipeline obj
		dataPipe = read_data.DataPipeline(filesList)
		selex_num = dataPipe.selex_num

		# Create and train the model

		net_model = Nets(selex_num=selex_num, model_name='multiple_nodes_net', validation=False)

		keras.utils.print_summary(net_model.model, line_length=None, positions=None, print_fn=None)

		#plot_model(net_model.model, to_file='model.png')
		net_model.train(dataPipe.train_generator, dataPipe.validation_generator, steps_per_epoch=100)
		predictions = net_model.test(dataPipe.test_generator)
		#predictions = None

		# Evaluate the model
		AUPR = util_functions.getAUPR(dataPipe.testData, predictions)
		resDict[str(sampleNum)] = round(AUPR, 5)

	print('########################################\n Results\n########################################')
	for k,v in resDict.items():
		print('Sample number {} : AUPR : {}'.format(k, v))
	print('Average AUPR : {}'.format(sum(resDict.values())/len(resDict)))

if __name__ == '__main__':
	_main()
