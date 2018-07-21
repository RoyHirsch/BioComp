import keras
import tensorflow as tf

import os
import random
from Utils.util_functions import f1
from keras.layers import Dense, Dropout, Flatten, Activation, multiply, Permute, Lambda, Add, Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import merge, Input, add
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model, Sequential
import keras.backend as K

class SimpleModel():

	def __init__(self, paramsDict, numOfModel):
		self._getModel(paramsDict, numOfModel)

	def _getModel(self, paramsDict, numOfModel):
		if numOfModel == 1:
			inputs = Input(shape=(36, 4))
			conv = Conv1D(256, 11, activation='relu')(inputs)
			max = (MaxPooling1D(6))(conv)
			drop = Dropout(0.25)(max)
			# conv2 = Conv1D(256, 11, activation='relu')(drop)
			# max2 = (MaxPooling1D(2))(conv2)
			# drop2 = Dropout(0.25)(max2)

			# squeez = Lambda(squeezLayer)(drop)
			# attention_probs = Dense(256, activation='softmax', name='attention_probs')(squeez)
			# sent_representation = multiply([squeez, attention_probs])
			# output = Dense(1, activation='sigmoid')(sent_representation)

			r_drop = Reshape((256, 4))(drop)
			attention_probs = Dense(1, activation='softmax', name='attention_probs')(r_drop)
			sent_representation = multiply([r_drop, attention_probs])

			sumLayer = Lambda(lambda x: K.sum(x, axis=1))(sent_representation)
			output = Dense(1, activation='sigmoid')(sumLayer)
			model = Model(inputs=inputs, outputs=output)
			model.compile(loss=keras.losses.binary_crossentropy,
			              optimizer=keras.optimizers.Adam(decay=paramsDict['lr_decay']),
			              metrics=['accuracy'])

			self.model = model

		elif numOfModel == 3:
			inputs = Input(shape=(36, 4))
			conv = Conv1D(256, 22, activation='relu')(inputs)
			max = (MaxPooling1D(8))(conv)
			drop = Dropout(0.25)(max)
			squeez = Lambda(squeezLayer)(drop)

			output = Dense(1, activation='sigmoid')(squeez)

			model = Model(inputs=inputs, outputs=output)
			model.compile(loss=keras.losses.binary_crossentropy,
			              optimizer=keras.optimizers.Adam(decay=0),
			              metrics=['accuracy'])

			self.model = model


		elif numOfModel == 2:
			input_shape = Input(shape=(36, 4))

			trail1 = Conv1D(paramsDict['depth'][0], 8, activation='relu', padding='same')(input_shape)
			trail1 = MaxPooling1D(paramsDict['max_pool'])(trail1)
			if paramsDict['dropout']:
				trail1 = Dropout(paramsDict['dropout'])(trail1)

			trail2 = Conv1D(paramsDict['depth'][1], 12, activation='relu', padding='same')(input_shape)
			trail2 = MaxPooling1D(paramsDict['max_pool'])(trail2)
			if paramsDict['dropout']:
				trail2 = Dropout(paramsDict['dropout'])(trail2)

			trail3 = Conv1D(paramsDict['depth'][2], 24, activation='relu', padding='same')(input_shape)
			trail3 = MaxPooling1D(paramsDict['max_pool'])(trail3)
			if paramsDict['dropout']:
				trail3 = Dropout(paramsDict['dropout'])(trail3)

			merged = keras.layers.concatenate([trail1, trail2, trail3], axis=2)
			merged = Flatten()(merged)

			fc = Dense(paramsDict['hidden_size'], activation='relu')(merged)
			if paramsDict['dropout']:
				fc = Dropout(paramsDict['dropout'])(fc)

			out = Dense(1, activation='sigmoid')(fc)

			model = Model(input_shape, out)

			if paramsDict['optimizer'] == 'adam':
				optimizer = keras.optimizers.Adam(decay=paramsDict['lr_decay'])
			elif paramsDict['optimizer'] == 'ada_delta':
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

# def attention_layer(input_dims):
# 	inputs = Input(shape=(input_dims,))
# 	attention_probs = Dense(input_dims, activation='softmax', name='attention_probs')(inputs)
# 	contex_vector = merge([inputs, attention_probs], name='attention_mul', mode='mul')


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
