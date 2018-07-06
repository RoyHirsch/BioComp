from model import Model
import numpy as np
import keras
from keras.layers import Flatten
from Utils import read_data
from Utils import util_functions

from keras.utils import plot_model

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

	# Create logger obj, print with the function : logging.info
	runFolderDir = util_functions.startLogging(isDump=False)

	# Create data pipeline obj
	dataPipe = read_data.DataPipeline(sys.argv)

	# Some simple model for test

	net_model = Nets(model_name='debug_net2',validation=False)

	keras.utils.print_summary(net_model.model, line_length=None, positions=None, print_fn=None)

	plot_model(net_model.model, to_file='model.png')
	net_model.train(dataPipe.train_generator, dataPipe.validation_generator, steps_per_epoch=10)
	predictions = net_model.test(dataPipe.test_generator)

	AUPR = util_functions.getAUPR(dataPipe.testData, predictions, False)

	layer = net_model.get_layerr(3)




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



# TF1_pbm.txt TF1_selex_0.txt TF1_selex_1.txt TF1_selex_2.txt TF1_selex_3.txt TF1_selex_4.txt