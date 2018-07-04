import keras
import os
import random
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import merge, Input, add
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model, Sequential

class SimpleModel():

	def __init__(self, paramsDict, numOfModel):
		self._getModel(paramsDict, numOfModel)

	def _getModel(self, paramsDict, numOfModel):
		if numOfModel == 1:
			model = Sequential()
			model.add(Conv1D(paramsDict['depth'], paramsDict['kernel_size'], activation='relu', input_shape=[36, 4]))
			model.add(MaxPooling1D(2))
			model.add(Dropout(0.25))
			# model.add(Conv1D(128, 13, activation='relu'))
			# model.add(MaxPooling1D(2))
			# model.add(Dropout(0.25))
			model.add(Flatten())
			model.add(Dense(paramsDict['hidden_size'], activation='relu'))
			model.add(Dropout(0.5))
			model.add(Dense(1, activation='sigmoid'))

			model.compile(loss=keras.losses.binary_crossentropy,
			              optimizer=keras.optimizers.Adam(decay=paramsDict['lr_decay']),
			              metrics=['accuracy'])

			self.model = model

		elif numOfModel == 2:
			input_shape = Input(shape=(36, 4))

			trail1 = Conv1D(paramsDict['depth'][0], 8, activation='relu', padding='same')(input_shape)
			trail1 = MaxPooling1D(2)(trail1)
			if paramsDict['dropout']:
				trail1 = Dropout(paramsDict['dropout'])(trail1)

			trail2 = Conv1D(paramsDict['depth'][1], 12, activation='relu', padding='same')(input_shape)
			trail2 = MaxPooling1D(2)(trail2)
			if paramsDict['dropout']:
				trail2 = Dropout(paramsDict['dropout'])(trail2)

			trail3 = Conv1D(paramsDict['depth'][2], 24, activation='relu', padding='same')(input_shape)
			trail3 = MaxPooling1D(2)(trail3)
			if paramsDict['dropout']:
				trail3 = Dropout(paramsDict['dropout'])(trail3)

			merged = keras.layers.concatenate([trail1, trail2, trail3], axis=2)
			merged = Flatten()(merged)

			fc = Dense(paramsDict['hidden_size'], activation='relu')(merged)
			if paramsDict['dropout']:
				fc = Dropout(paramsDict['dropout'])(fc)
			out = Dense(1, activation='sigmoid')(fc)

			model = Model(input_shape, out)

			model.compile(loss=keras.losses.binary_crossentropy,
			              optimizer=keras.optimizers.Adam(decay=paramsDict['lr_decay']),
			              metrics=['accuracy'])

			self.model = model

	def train(self, tain_generator, validation_generator, n_epochs, n_workers):
		self.model.fit_generator(generator=tain_generator, validation_data=validation_generator, steps_per_epoch=2000, epochs=n_epochs,
		                         validation_steps=7000, use_multiprocessing=n_workers!=0, workers=n_workers,verbose=1)


	def predict(self, test_generator, n_workers):
		predictions = self.model.predict_generator(generator=test_generator, use_multiprocessing=n_workers!=0,
		                                           workers=n_workers, verbose=0)
		return predictions

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
			'depth': [[20,20,24],[40,40,48],[80,80,96]],
			'dropout': [0, 0.25, 0.5],
			'hidden_size': [32, 64, 128],
			'lr_decay': [0, 1e-6, 1e-7]
		}

		paramsDict = {}
		paramsDict['depth'] = random.choice(mainDict['depth'])
		paramsDict['dropout'] = random.choice(mainDict['dropout'])
		paramsDict['hidden_size'] = random.choice(mainDict['hidden_size'])
		paramsDict['lr_decay'] = random.choice(mainDict['lr_decay'])
		return paramsDict


def printDict(paramsDict):
	print('Params are :')
	for key, value in paramsDict.items():
		print('{} : {}'.format(key, value))

