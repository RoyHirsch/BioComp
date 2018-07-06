
import numpy as np
import keras
import os
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import merge, Input, add
from keras.optimizers import SGD, Adam
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
        self.optimizer_type = parameters["optimizer"]
        self.learning_rate_decay_rate = parameters["learning_rate_decay_rate"]

    def add_conv_layer(self, input_, kernel_size, filters, dropout=True, batch_norm=True, padding='valid'):

        """Add to exist model basic layer according to user specifications
        kernel size didn't added to the parameters so there will be flexibility in model build"""

        output_ = Conv1D(filters=filters, kernel_size=kernel_size, strides=self.strides,
                         activation=self.activation1,
                         kernel_regularizer=l2(self.regularization_coeff), padding=padding)(input_)

        # TODO consider add regularization to external params. consider use of other weigths initializer
        if batch_norm:
            output_ = BatchNormalization()(output_)

        output_ = MaxPooling1D(pool_size=self.pool_size, strides=None)(output_)
        if dropout:
            output_ = Dropout(self.dropout)(output_)

        return output_

    def add_fc_layer(self, input_, size, dropout=False, batch_norm=False, no_activation=False):

         # TODO consider use of other weigths initializer
        if no_activation:
            activation = None
        else:
            activation = self.fc_activation
        output_ = Dense(size, activation=activation)(input_)
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


            if dropout:
                conv_node = Dropout(self.dropout)(conv_node)

        conv_node = MaxPooling1D(pool_size=self.pool_size, strides=None)(conv_node)
        skip_node = MaxPooling1D(pool_size=self.pool_size, strides=None)(skip_node)

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


    def create_multiple_conv_layer_model(self, input_, kernel_size=None,padding='valid'):
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
                                                  batch_norm=False, padding=padding)
                if layer == 'residual':
                    output_ = self.add_residual_block(output_, kernel_size=kernel_size, filters=filters[rep],
                                                      dropout=True, batch_norm=False,)
        return output_

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

    def create_multiple_nodes_model(self, input_=None):

        layers_dict = self.conv_layers_dict
        nodes_dict = self.nodes_dict

        nodes_num = nodes_dict["nodes"]
        kernel_sizes = nodes_dict["kernel_sizes"]

        if len(kernel_sizes) != nodes_num:
            raise ValueError('Length of kernel sizes list is not equal with number of nodes ')

        nodes = []
        for node in range(nodes_num):
            nodes.append(self.create_multiple_conv_layer_model(input_=input_,
                                                               kernel_size=kernel_sizes[node], padding='same'))

        max_dim = len(np.shape(nodes[0])) - 1
        merged = keras.layers.concatenate(nodes, axis=max_dim)
        output_ = merged
        return output_


class Nets(BuildModel):

    def model_dict(self, model_name):
        return {
            'base_net': self.base_net,
            'res_net': self.res_net,
            'multiple_nodes_net': self.multiple_nodes_net,
            'debug_net': self.debug_net,
            'debug_net2': self.debug_net2,


        }.get(model_name)

    def __init__(self, model_name='base_net', validation=True):
        super(Nets, self).__init__(validation=validation)
        self.model = self.model_dict(model_name)()

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
        output_ = self.add_conv_layer(input_=input_, kernel_size=3, filters=128, dropout=True, batch_norm=False)
        output_ = self.add_conv_layer(input_=output_, kernel_size=3, filters=64, dropout=True, batch_norm=False)
        output_ = self.add_conv_layer(input_=output_, kernel_size=3, filters=32, dropout=True, batch_norm=False)
        output_ = Flatten()(output_)
        output_ = self.add_fc_layer(input_=output_, size=128)
        output_ = self.add_fc_layer(input_=output_, size=1, no_activation=True)
        output_ = self.add_sigmoid_layer(output_)
        net = Model(inputs=input_, outputs=output_)
        return net

    def debug_net2(self):

        input_ = Input(shape=self.input_shape)
        output_ = Conv1D(filters=32,kernel_size=3,strides=1)(input_)
        output_ = Dense(1)(output_)
        net = Model(inputs=input_, outputs=output_)
        return net


    def res_net(self):
        input_ = Input(shape=self.input_shape)
        output_ = self.create_multiple_conv_layer_model(input_=input_)
        output_ = Flatten()(output_)
        output_ = self.add_fc_layer(input_=output_, size=32)
        output_ = self.add_fc_layer(input_=output_, size=1, no_activation=True)
        output_ = self.add_sigmoid_layer(output_)
        net = Model(inputs=input_, outputs=output_)

        return net


    def multiple_nodes_net(self):
        input_ = Input(shape=self.input_shape)
        output_ = self.create_multiple_nodes_model(input_=input_)
        output_ = self.add_fc_layer(output_, 2, dropout=True, no_activation=True)
        output_ = Activation(activation='softmax')(output_)
        net = Model(inputs=input_, outputs=output_)
        return net

    def train(self, training_generator, validation_generator, steps_per_epoch=None):
        self.model.compile(loss=self.loss, optimizer=self.optimizer(), metrics=['accuracy'])
        if not self.validation:
            validation_generator = None
        self.model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            use_multiprocessing=True,
                            workers=6,
                            verbose=1, steps_per_epoch=steps_per_epoch)

    def test(self, test_generator):
        model = self.model
        predictions = model.predict_generator(generator=test_generator, use_multiprocessing=True, workers=6, verbose=1)
        return predictions

    def optimizer(self):
        if self.optimizer_type == 'Adam':
            return Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None,
                        decay=self.learning_rate_decay_rate, amsgrad=False)
        if self.optimizer_type == 'SGD':
            return SGD(lr=self.learning_rate, decay=self.learning_rate_decay_rate,
                       momentum=self.momentum, nesterov=True)


    def get_layerr(self, layer_num):
        layer = self.model.get_layer(index=layer_num)
        return layer

















