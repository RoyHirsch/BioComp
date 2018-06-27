from model import Model
import numpy as np
import keras
from keras.layers import Flatten
from src.Utils import read_data

# Generate dummy data - temporally
x_train = np.random.random((100, 36, 4))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 36, 4))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

# Data load

debugPath = '/Users/royhirsch/Documents/GitHub/BioComp/train_data/'
dataObj = read_data.DataPipeline(dataRoot=debugPath, mode='Train', argsDist={})
train_loader = read_data.DataGenerator
test_loader = read_data.TestDataGenerator

training_generator = train_loader(dataObj.trainData, dataObj.trainLabel)
validation_generator = train_loader(dataObj.validationData, dataObj.validationLabel)
test_generator = test_loader(dataObj.testData)


model = Model()
model.add_conv_layer(kernel_size=3, dropout=True, batch_norm=True, input_layer=True)
model.add_conv_layer(kernel_size=6, dropout=True, batch_norm=True)
model.model.add(Flatten())
model.add_fc_layer(10)
model.gen_train(x_train, y_train)
res = model.gen_test(x_test)



