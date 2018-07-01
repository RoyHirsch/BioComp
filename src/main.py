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

# Constants #
DEBUG = True

def _main():

	if (len(sys.argv)) < 3:
		print("Missing arguments, call should be : python main.py <pbm> <selex0> <selex1> <selex2> ... <selex5>")
		exit(1)

	runFolderDir = util_functions.startLogging(isDump=False)
	# logging.info

	dataPipe = read_data.DataPipeline(sys.argv, argsDict={'batch_size': 32,
	                                                      'dim': (36,4),
	                                                      'n_channels': 1,
	                                                      'n_classes': 1})

	# Some simple model for test
	# conv-conv-flat-dense
	net_model = BuildModel()
	net_model.train(dataPipe.train_generator, dataPipe.validation_generator)
	predictions = net_model.test(dataPipe.test_generator)

	AUC = util_functions.getAUPR(dataPipe.testData, predictions, )

if __name__ == '__main__':
	_main()

'''
modifications for smaller test:
TestDataGenerator - batch size

model.fit_generator - steps_per_epoch

predict_generator - steps

read_data selexFilesPathList[:1]
'''