import keras
import tensorflow as tf
import os
import random
from keras.layers import Dense, Dropout, Flatten, Input, Conv1D, MaxPooling1D, concatenate
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.models import Model, Sequential
from Utils.util_functions import get_model_parameters

########################
# LOAD PARAMS
# Reads external json file with parameters.
########################
params_file_name = os.path.abspath(__file__ + '/../') + '/Utils/config.json'
parameters = get_model_parameters(params_file_name)

#########################################################################
# Description: Creates the neural network model to be trained and evaluated.
#
# Input: numOfModel
# Output: Model obj
#########################################################################
class NetModel():

	def __init__(self, numOfModel):
		self._getModel(numOfModel)

	def _getModel(self, numOfModel):

		# Model 1: simple convectional model.
		# Inspired by DeepBind
		if numOfModel == 1:
			inputs = Input(shape=(36, 4))
			conv = Conv1D(256, 11, activation='relu')(inputs)
			max = (MaxPooling1D(6))(conv)
			drop = Dropout(0.25)(max)
			fc = Dense(1, activation='relu')(drop)
			output = Dense(1, activation='sigmoid')(fc)
			model = Model(inputs=inputs, outputs=output)
			model.compile(loss=keras.losses.binary_crossentropy,
			              optimizer=keras.optimizers.Adam(decay=parameters['lr_decay']),
			              metrics=['accuracy'])

			self.model = model

		# Model 2: three-way convectional model.
		# Calculates three parallel trails, each searching for different size motif length.
		elif numOfModel == 2:
			inputs = Input(shape=(36, 4))

			trail1 = Conv1D(parameters['depth'][0], 8, activation='relu', padding='same')(inputs)
			trail1 = MaxPooling1D(parameters['max_pool'])(trail1)
			if parameters['dropout']:
				trail1 = Dropout(parameters['dropout'])(trail1)

			trail2 = Conv1D(parameters['depth'][1], 12, activation='relu', padding='same')(inputs)
			trail2 = MaxPooling1D(parameters['max_pool'])(trail2)
			if parameters['dropout']:
				trail2 = Dropout(parameters['dropout'])(trail2)

			trail3 = Conv1D(parameters['depth'][2], 24, activation='relu', padding='same')(inputs)
			trail3 = MaxPooling1D(parameters['max_pool'])(trail3)
			if parameters['dropout']:
				trail3 = Dropout(parameters['dropout'])(trail3)

			merged = concatenate([trail1, trail2, trail3], axis=2)
			merged = Flatten()(merged)

			fc = Dense(parameters['hidden_size'], activation='relu')(merged)
			if parameters['dropout']:
				fc = Dropout(parameters['dropout'])(fc)

			output = Dense(1, activation='sigmoid')(fc)

			model = Model(inputs, output)

			if parameters['optimizer'] == 'adam':
				optimizer = keras.optimizers.Adam(decay=parameters['lr_decay'])

			elif parameters['optimizer'] == 'ada_delta':
				optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

			model.compile(loss=keras.losses.binary_crossentropy,
			              optimizer=optimizer,
			              metrics=['accuracy'])

			self.model = model

	def train(self, tain_generator, validation_generator, steps_per_epoch, n_epochs, n_workers):
		self.model.fit_generator(generator=tain_generator, validation_data=validation_generator, steps_per_epoch=steps_per_epoch, epochs=n_epochs,
		                         use_multiprocessing=n_workers!=0, workers=n_workers,verbose=2)

	def predict(self, test_generator, n_workers):
		predictions = self.model.predict_generator(generator=test_generator, use_multiprocessing=n_workers!=0,
		                                           workers=n_workers, verbose=0)
		return predictions


#########################################################################
# Description: Helper function for parameters search.
#              Generates randomly selected permutations of hyper-params.
#              The function was used during the model development stage.
#
# Input: numOfModel
# Output: paramsDict
#########################################################################
def getParamsDict(numOfModel):

	if numOfModel==1:
		mainDict = {
			'depth'      : [128, 256, 512],
			'kernel_size': [11, 13, 15],
			'hidden_size': [32, 64, 128],
			'lr_decay'   : [0, 1e-6, 1e-7]
		}

		paramsDict = {}
		paramsDict['depth'] =random.choice(mainDict['depth'])
		paramsDict['kernel_size'] = random.choice(mainDict['kernel_size'])
		paramsDict['hidden_size'] = random.choice(mainDict['hidden_size'])
		paramsDict['lr_decay'] = random.choice(mainDict['lr_decay'])
		return paramsDict

	elif numOfModel == 2:
		mainDict = {
			'depth': [[40,40,48],[80,80,96]],
			'dropout': [0, 0.25, 0.5],
			'hidden_size': [32, 64, 128],
			'lr_decay': [0, 1e-6, 1e-7],
			'max_pool': [2, 4, 6],
			'optimizer': ['adam', 'ada_delta']
		}

		paramsDict = {}
		paramsDict['depth'] = random.choice(mainDict['depth'])
		paramsDict['dropout'] = random.choice(mainDict['dropout'])
		paramsDict['hidden_size'] = random.choice(mainDict['hidden_size'])
		paramsDict['lr_decay'] = random.choice(mainDict['lr_decay'])
		paramsDict['max_pool'] = random.choice(mainDict['max_pool'])
		paramsDict['optimizer'] = random.choice(mainDict['optimizer'])
		return paramsDict


def printDict(paramsDict):
	print('The hyper-params are :')
	for key, value in paramsDict.items():
		print('{} : {}'.format(key, value))
