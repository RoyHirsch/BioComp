from src.Utils import parameters, read_data
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

from src.Utils import parameters, read_data

# load parameters
params_file_name = './src/Utils/config.json'
parameters = parameters(params_file_name)


class Model:

    def __init__(self):
        self.activation1 = parameters["activation1"]
        self.activation2 = parameters["activation2"]
        self.conv_filters = parameters["conv_filters"]
        self.strides = parameters["strides"]   # TODO add to Tuple format
        self.pool_size = parameters["pool_size"]
        self.dropout = parameters["dropout"]


        # Todo - add more model parameters\

        # initialize model
        self.model = Sequential()

    def basic_conv_layer_add(self, kernel_size, dropout=True, batch_norm=True):

        model = self.model

        """Add to exist model basic layer according to user specifications
        kernel size didn't added to the parameters so there will be flexibility in model build"""

        model.add(Conv2D(filters=self.conv_filters, kernel_size=kernel_size, strides=self.strides,
                         activation=self.activation1, input_shape=(100, 100, 3)))
        if batch_norm:
            model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=self.pool_size, strides=None))
        if dropout:
            model.add(Dropout(self.dropout))



