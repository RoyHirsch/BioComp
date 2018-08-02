import os
import sys
sys.path.append(os.path.realpath(__file__ + "/../../"))
from Utils import read_data, util_functions
from simple_model import NetModel




def _main():

	if (len(sys.argv)) < 3:
		print("Missing arguments, call should be : python main.py <pbm> <selex0> <selex1> <selex2> ... <selex5>")
		exit(1)


	# Reads external json file with parameters.
	params_file_name = os.path.abspath(__file__ + '/../') + '/Utils/config.json'
	parameters = util_functions.get_model_parameters(params_file_name)

	# Create data pipeline obj
	dataPipe = read_data.DataPipeline(sys.argv)

	model = NetModel(numOfModel=2,input_shape=(dataPipe.input_shape))
	model.train(tain_generator=dataPipe.train_generator,
	            validation_generator=None,
	            steps_per_epoch=parameters['steps_per_epoch'],
	            n_epochs=parameters['n_epochs'],
	            n_workers=parameters['n_workers'])

	predictions = model.predict(dataPipe.test_generator, parameters['n_workers'])
	AUPR = util_functions.getAUPR(dataPipe.testData, predictions)
	print(AUPR)

	# Sort and save fo files the PBM
	sortedPBMarray = util_functions.sortPBM(dataPipe.testData, predictions)
	TF_index = dataPipe.TF_index
	util_functions.dumpPBMtoFile(os.path.realpath(__file__ + "/../"), 'sortedPBM_TF' + str(TF_index), sortedPBMarray)

if __name__ == '__main__':
	_main()
