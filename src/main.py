
from model import BuildModel
from Utils.read_data import DataPipeline, DataGenerator, TestDataGenerator


def main():

    # Load data
    debugPath = '/Users/amitzeligman/Downloads/train'
    dataObj = DataPipeline(dataRoot=debugPath, mode='debug', argsDist={})
    train_loader = DataGenerator
    test_loader = TestDataGenerator
    training_generator = train_loader(dataObj.trainData, dataObj.trainLabel)
    validation_generator = train_loader(dataObj.validationData, dataObj.validationLabel)
    test_generator = test_loader(dataObj.testData)

    net_model = BuildModel()
    net_model.train(training_generator, validation_generator)
    pred = net_model.test(test_generator)


if __name__ == '__main__':
    main()

