from model import Model
import numpy as np
import keras
from keras.layers import Flatten

# Generate dummy data - temporally
x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)


model = Model()
model.add_conv_layer(kernel_size=(3, 3), dropout=True, batch_norm=True, input_layer=True)
model.add_conv_layer(kernel_size=(6, 6), dropout=True, batch_norm=True)
model.model.add(Flatten())
model.add_fc_layer(256)
model.train(x_train, y_train)
model.test(x_test)



