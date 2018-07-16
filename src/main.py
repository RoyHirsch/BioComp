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
from simple_model import SimpleModel


def _main():

	if (len(sys.argv)) < 3:
		print("Missing arguments, call should be : python main.py <pbm> <selex0> <selex1> <selex2> ... <selex5>")
		exit(1)

	# Create logger obj, print with the function : logging.info
	runFolderDir = util_functions.startLogging(isDump=True)

	# Create data pipeline obj
	dataPipe = read_data.DataPipeline(sys.argv)

	model = SimpleModel()
	model.train(dataPipe.train_generator, dataPipe.validation_generator, 1, 6)
	predictions = model.predict(dataPipe.test_generator,6)

	# Some simple model for test
	# net_model = BuildModel()
	# predictions = net_model.test(dataPipe.test_generator)
	# net_model.train(dataPipe.train_generator, dataPipe.validation_generator)
	# predictions = net_model.test(dataPipe.test_generator)

	# Evaluate the model
	AUPR = util_functions.getAUPR(dataPipe.testData, predictions)

if __name__ == '__main__':
	_main()

'''
RoyH 0107
---------
modifications for smaller net test:
TestDataGenerator - batch size

model.fit_generator - steps_per_epoch

predict_generator - steps

read_data selexFilesPathList[:1]
'''