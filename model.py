
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, Conv1D, MaxPooling1D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

from src.Utils.parameters import parameters
#from src.Utils.read_data import DataPipeline,DataGenerator


# load parameters
params_file_name = './src/Utils/config.json'
parameters = parameters(params_file_name)


class Model:

    def __init__(self):
        self.input_shape = tuple(parameters["input_shape"])
        self.activation1 = parameters["activation1"]
        self.activation2 = parameters["activation2"]
        self.conv_filters = parameters["conv_filters"]
        self.strides = tuple(parameters["strides"])
        self.pool_size = tuple(parameters["pool_size"])
        self.dropout = parameters["dropout"]
        self.fc_activation = parameters["fc_activation"]
        self.learning_rate = parameters["learning_rate"]
        self.momentum = parameters["momentum"]
        self.loss = parameters["loss"]
        self.batch_size = parameters["batch_size"]
        self.epochs = parameters["epochs"]
        self.regularization_coeff = parameters["regularization_coeff"]

        # initialize model
        self.model = Sequential()

    def add_conv_layer(self, kernel_size, dropout=True, batch_norm=True, input_layer=False):

        model = self.model

        """Add to exist model basic layer according to user specifications
        kernel size didn't added to the parameters so there will be flexibility in model build"""
        if input_layer:
            model.add(Conv1D(filters=self.conv_filters, kernel_size=kernel_size, strides=self.strides,
                             activation=self.activation1, input_shape=self.input_shape,
                             kernel_regularizer=l2(self.regularization_coeff)))
        else:
            model.add(Conv1D(filters=self.conv_filters, kernel_size=kernel_size, strides=self.strides,
                             activation=self.activation1, kernel_regularizer=l2(self.regularization_coeff)))

        # TODO consider add regularization to external params. consider use of other weigths initializer
        if batch_norm:
            model.add(BatchNormalization())

        model.add(MaxPooling1D(pool_size=self.pool_size, strides=None))
        if dropout:
            model.add(Dropout(self.dropout))

    def add_fc_layer(self, size, dropout=False, batch_norm=False):

        model = self.model     # TODO consider use of other weigths initializer
        model.add(Dense(size, activation=self.fc_activation))
        if batch_norm:
            model.add(BatchNormalization())
        if dropout:
            model.add(Dropout(self.dropout))

    def optimizer(self):

        return SGD(lr=self.learning_rate, decay=1e-6, momentum=self.momentum, nesterov=True)

    def train(self, data, labels, ):
        model = self.model
        sgd = self.optimizer()
        model.compile(loss=self.loss, optimizer=sgd)
        model.fit(data, labels, batch_size=self.batch_size, epochs=self.epochs,
                  validation_split=0.1, validation_data=None, validation_steps=None)
        # TODO - add validation handling

    def test(self, test_data, test_labels):

        model = self.model
        results = model.evaluate(test_data, test_labels, batch_size=32)
        return results

    def gen_train(self, training_generator, validation_generator):
        model = self.model
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            use_multiprocessing=True,
                            workers=6,
                            verbose=1)

    def gen_test(self, test_generator):
        model = self.model
        predictions = model.predict_generator(generator=test_generator)
        return predictions





