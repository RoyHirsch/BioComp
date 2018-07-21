import sys
import os
sys.path.append(os.path.realpath(__file__ + "/../../"))
from Utils import read_data, util_functions
from simple_model import SimpleModel


def _main():

	if (len(sys.argv)) < 3:
		print("Missing arguments, call should be : python main.py <pbm> <selex0> <selex1> <selex2> ... <selex5>")
		exit(1)

	# Create data pipeline obj
	dataPipe = read_data.DataPipeline(sys.argv)

	mainDict = {
		'depth': [80, 80, 96],
		'dropout': 0,
		'hidden_size': 32,
		'lr_decay': 0
	}

	model = SimpleModel(paramsDict=mainDict, numOfModel=2)
	model.train(tain_generator=dataPipe.train_generator, validation_generator=None,
	            steps_per_epoch=10000, n_epochs=3, n_workers=6)

	predictions = model.predict(dataPipe.test_generator, 6)
	AUPR = util_functions.getAUPR(dataPipe.testData, predictions)
	print(AUPR)

if __name__ == '__main__':
	_main()
