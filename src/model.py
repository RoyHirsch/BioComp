
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
        self.conv_layers_dict = parameters["conv_layers_dict"]
        self.fc_layers_dict = parameters["fc_layers_dict"]
        self.skip_connection_num = parameters["skip_connection_num"]
        self.nodes_dict = parameters["nodes_dict"]

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

    def add_residual_block(self, input_, kernel_size, filters, dropout=True, batch_norm=True):
        skip_node = input_
        conv_node = input_
        n_skip = self.skip_connection_num
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

    @staticmethod
    def add_sigmoid_layer(input_):

        output_ = Activation('sigmoid')(input_)
        return output_


class BuildModel(ModelFunctions):

    def __init__(self, validation=True):
        super(BuildModel, self).__init__()
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

    def create_multiple_conv_layer_model(self, input_layer=False, input_=None, kernel_size=None):
        if input_layer:
            output_ = Input(shape=self.input_shape)
        else:
            output_ = input_
        layers_dict = self.conv_layers_dict
        layers = layers_dict['layers']
        reps = layers_dict['reps']
        filters = layers_dict['filters']
        if kernel_size is None:
            kernel_size = layers_dict['kernel_size']

        if len(filters) != reps:
            raise ValueError('Length of filters list is not equal with number of reps ')

        for rep in range(reps):

            for layer in layers:
                if layer == 'conv':
                    output_ = self.add_conv_layer(output_, kernel_size=kernel_size, filters=filters[rep], dropout=True,
                                                  batch_norm=False)
                if layer == 'reidual':
                    output_ = self.add_residual_block(output_, kernel_size=kernel_size, filters=filters[rep],
                                                      dropout=True, batch_norm=False,)

    def create_multiple_fc_layer_model(self, input_layer=False, input_=None, sizes=None):
        if input_layer:
            output_ = Input(shape=self.input_shape)
        else:
            output_ = input_
        layers_dict = self.fc_layers_dict
        if sizes is None:
            sizes = layers_dict["sizes"]

        for rep in range(len(sizes)):
            output_ = self.add_fc_layer(output_, sizes[rep], dropout=False, BatchNormalization=True)
            return output_

    def create_multiple_nodes_model(self, input_layer=True, input_=None):

        if input_layer:
            output_ = Input(shape=self.input_shape)
        else:
            output_ = input_

        layers_dict = self.layers_dict
        nodes_dict = self.nodes_dict

        nodes_num = nodes_dict["nodes"]
        kernel_sizes = nodes_dict["kernel_sizes"]

        if len(kernel_sizes) != nodes_num:
            raise ValueError('Length of kernel sizes list is not equal with number of nodes ')

        nodes = []
        for node in range(nodes_num):
            nodes.append(self.create_multiple_conv_layer_model(layers_dict, input_=input_, kernel_size=kernel_sizes[node]))
        merged = keras.layers.concatenate(nodes, axis=1)
        output_ = merged
        return output_


class Nets(BuildModel):

    def model_dict(self, model_name):
        return {
            #'base_net': self.base_net(),
            #'res_net': self.res_net(),
            #'multiple_nodes_res_net': self.multiple_nodes_res_net(),
            'debug_net': self.debug_net(),

        }.get(model_name, self.base_net())

    def __init__(self, model_name='base_net', validation=True):
        super(Nets, self).__init__(validation=validation)
        self.model = self.model_dict(model_name)

    def base_net(self):
        input_ = Input(shape=self.input_shape)
        output_ = self.add_conv_layer(input_=input_, kernel_size=3, filters=self.conv_filters, dropout=True,
                                      batch_norm=True)
        output_ = self.add_residual_block(input_=output_, kernel_size=3, filters=self.conv_filters)
        output_ = Flatten()(output_)
        output_ = self.add_fc_layer(input_=output_, size=1)
        output_ = self.add_sigmoid_layer(input_=output_)
        net = Model(inputs=input_, outputs=output_)
        return net

    def debug_net(self):

        input_ = Input(shape=self.input_shape)
        output_ = self.add_conv_layer(input_=input_, kernel_size=7, filters=32)
        output_ = self.add_conv_layer(input_=output_, kernel_size=7, filters=64)
        output_ = Flatten()(output_)
        output_ = self.add_fc_layer(input_=output_, size=128)
        output_ = self.add_fc_layer(input_=output_, size=1)
        net = Model(inputs=input_, outputs=output_)
        return net

    def res_net(self):
        input_ = Input(shape=self.input_shape)
        output_ = self.create_multiple_conv_layer_model(input_=input_)
        net = Model(inputs=input_, outputs=output_)
        # TODO finish the model
        return net

    def multiple_nodes_res_net(self):
        pass















