
import numpy as np
import keras
import os
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import merge, Input, add
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model

from Utils.parameters import parameters

# load parameters
params_file_name = os.path.abspath(__file__ + '/../') + '/Utils/config.json'
parameters = parameters(params_file_name)


class ModelFunctions:

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

    def add_conv_layer(self, input_, kernel_size, filters, dropout=True, batch_norm=True):

        """Add to exist model basic layer according to user specifications
        kernel size didn't added to the parameters so there will be flexibility in model build"""

        output_ = Conv1D(filters=filters, kernel_size=kernel_size, strides=self.strides,
                         activation=self.activation1,
                         kernel_regularizer=l2(self.regularization_coeff))(input_)

        # TODO consider add regularization to external params. consider use of other weigths initializer
        if batch_norm:
            output_ = BatchNormalization()(output_)

        output_ = MaxPooling1D(pool_size=self.pool_size, strides=None)(output_)
        if dropout:
            output_ = Dropout(self.dropout)(output_)

        return output_

    def add_fc_layer(self, input_, size, dropout=False, batch_norm=False):

         # TODO consider use of other weigths initializer
        output_ = Dense(size, activation=self.fc_activation)(input_)
        if batch_norm:
            output_ = BatchNormalization()(output_)
        if dropout:
            output_ = Dropout(self.dropout)(output_)

        return output_

    def add_residual_block(self, input_, n_skip, kernel_size, filters, dropout=True, batch_norm=True):
        skip_node = input_
        conv_node = input_
        for i in range(n_skip):
            conv_node = Conv1D(filters=filters, kernel_size=kernel_size, strides=self.strides,
                               activation=self.activation1, padding='same',
                               kernel_regularizer=l2(self.regularization_coeff))(conv_node)
            skip_node = Conv1D(filters=filters, kernel_size=[1],
                               strides=self.strides, padding='same')(skip_node)

            if batch_norm:
                conv_node = BatchNormalization()(conv_node)
                skip_node = BatchNormalization()(skip_node)

            conv_node = MaxPooling1D(pool_size=self.pool_size, strides=None)(conv_node)
            skip_node = MaxPooling1D(pool_size=self.pool_size, strides=None)(skip_node)

            if dropout:
                conv_node = Dropout(self.dropout)(conv_node)
        output_ = add([skip_node, conv_node])

        return output_

    def add_sigmoid_layer(self, input_):

        output_ = Activation('sigmoid')(input_)
        return output_


class BuildModel(ModelFunctions):

    def model_dict(self, mode_name):
        return {
            'base_net': self.base_net(),

            # TODO add more archs here
        }.get(mode_name, self.base_net())

    def __init__(self, validation=True, model_name='base_net'):
        super(BuildModel, self).__init__()

        self.model = self.model_dict(model_name)
        self.learning_rate = parameters["learning_rate"]
        self.momentum = parameters["momentum"]
        self.validation = validation

    def train(self, training_generator, validation_generator, steps_per_epoch=None):
        model = self.model
        sgd = self.optimizer()
        model.compile(loss=self.loss, optimizer=sgd)
        if not self.validation:
            validation_generator = None
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            use_multiprocessing=True,
                            workers=6,
                            verbose=1, steps_per_epoch=steps_per_epoch)

    def test(self, test_generator):
        model = self.model
        predictions = model.predict_generator(generator=test_generator, use_multiprocessing=True, workers=6, verbose=1)
        return predictions

    def optimizer(self):

        return SGD(lr=self.learning_rate, decay=1e-6, momentum=self.momentum, nesterov=True)

    def base_net(self):
        input_ = Input(shape=self.input_shape)
        output_ = self.add_conv_layer(input_=input_, kernel_size=3, filters=self.conv_filters, dropout=True, batch_norm=True)
        output_ = self.add_residual_block(input_=output_, n_skip=1, kernel_size=3, filters=self.conv_filters)
        output_ = Flatten()(output_)
        output_ = self.add_fc_layer(input_=output_, size=1)
        output_ = self.add_sigmoid_layer(input_=output_)
        net = Model(inputs=input_, outputs=output_)
        return net

    def create_multiple_layer_model(self, layers_dict):
        output_ = Input(shape=self.input_shape)

        # dummy dict for debug
        layers_dict = {'layers': [ 'conv', 'residual'],
                       'reps': 4,
                       'filters': [256, 128, 64, 32, 16],
                       'kernel_size': 3
                       }

        layers = layers_dict["layers"]
        reps = layers_dict['rep']
        filters = layers_dict['filters']
        kernel_size = layers_dict["kernel_size"]

        for rep in range(reps):

            for index, layer in enumerate(layers):
                if layer == 'conv':
                    output_ = self.add_conv_layer(output_, kernel_size=kernel_size,filters=filters[index], dropout=True,
                                                  batch_norm=False)
                if layer == 'reidual':
                    # TODO decide about uniform parameters choose
                    self.add_residual_block(output_, n_skip=2, kernel_size=kernel_size, filters=filters[index],
                                            dropout=True, batch_norm=False,)

















    # TODO make conv filters as iterable list
    # TODO exposure variables out - steps, validation(true/false) etc....









