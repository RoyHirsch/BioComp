import sys
import os
sys.path.append(os.path.realpath(__file__ + "/../../"))
from Utils import read_data, util_functions
from simple_model import SimpleModel
from simple_model import *
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

######################
# CONSTANTS
######################
numOfRuns = 1
modelNum = 2
num_of_tries = 2

def _main(numOfRuns=2):


	yaronAUPR = []
	ini_listOfSamples = [57, 59, 86, 80, 41, 54, 9, 108, 73, 30, 62, 96, 5, 50, 74, 24, 99, 101, 68, 31]
	ini_yaronAUPR = [0.594610698,0.514773825,0.4717869,0.341958113,0.250109638,0.246641534,0.239458146,0.237103258,
				 0.221448961,0.173986496,0.172595686,0.167862911,0.114530832,0.113215591,0.103344651,0.002081527,
				 0.002065405,0.002049701,0.087953909,0.037662088]

	listOfSamples = np.random.choice(ini_listOfSamples,numOfRuns)
	for i in range(len(listOfSamples)):
		yaronAUPR.append(ini_yaronAUPR[ini_listOfSamples.index(listOfSamples[i])])
	print('################')
	print('samples_number: %s' % listOfSamples)
	print('Yaron AUPR:')
	print(yaronAUPR)
	print('################')
	# Create logger obj, print with the function : logging.info
	runFolderDir = util_functions.startLogging(isDump=True)

	# Permutate paramsDict to
	paramsDict = getParamsDict(modelNum)
	
	# Run the nn over num of samples


	for i in range(num_of_tries):
		avg_weigths = np.abs(np.random.randn(3))+1
		avg_weigths = np.sort(avg_weigths)
		avg_weigths = avg_weigths[::-1]
		ind = 0
		resDict = {}
		avg_AUPR = np.zeros(numOfRuns)
		top_AUPR = np.zeros(numOfRuns)
		AUPR = np.zeros([numOfRuns, 6])
		predictions = []

		for sample in range(numOfRuns):
			print('++++++++++++ sample number %i: ++++++++++++' % sample)
			# Get random sample data and export the files as list
			sampleNum, filesList = util_functions.getTrainSampleFromList(dataRoot=os.path.realpath(__file__ + '/../../')+'/train',
		                                                             listOfSamples=listOfSamples,
		                                                             ind=ind)
			ind += 1
			selex_num = len(filesList) - 3
			selex_index = np.linspace(1, selex_num, selex_num)
			selex = 1
			dataPipe = read_data.DataPipeline(filesList, int(selex))
			test_length = len(dataPipe.testData)
			tmp_predictions = np.zeros([selex_num, test_length])
			for selex in selex_index:
				selex = int(selex)
				print('+++++++++ selex number %i +++++++++' % selex)
				if selex != 1:
					# Create data pipeline obj
					dataPipe = read_data.DataPipeline(filesList, selex)
	
				# Create and train the model
				model = SimpleModel(paramsDict, modelNum)
				model.train(tain_generator=dataPipe.train_generator, validation_generator=None,
			            steps_per_epoch=1, n_epochs=1, n_workers=6)

				# Evaluate the model
				tmp_predictions[selex-1,:] = np.squeeze(model.predict(dataPipe.test_generator, 6))
				#predictions = model.predict(dataPipe.test_generator, 6)
				print('selex %i AUPR:' % selex)
				AUPR[sample,selex-1]=util_functions.getAUPR(dataPipe.testData, tmp_predictions[selex-1,:])

			predictions.append(tmp_predictions)
			weights = np.linspace(1,selex_num/2,selex_num)
			avg_predictions = np.average(predictions[sample], axis=0,weights=weights)
			top_predictions = (predictions[sample][selex_num-1,:]*avg_weigths[0] + predictions[sample][selex_num-2,:]*avg_weigths[1] +
							   predictions[sample][selex_num-3,:]*avg_weigths[2])/(np.sum(avg_weigths))
			print('average AUPR:')
			avg_AUPR[sample] = util_functions.getAUPR(dataPipe.testData, avg_predictions)
			print('top AUPR:')
			top_AUPR[sample] = util_functions.getAUPR(dataPipe.testData, top_predictions)

			resDict[str(sampleNum)] = round(avg_AUPR[sample], 5)
		print('\n########################################\n Results\n########################################\n')
		print('weigths:')
		print(avg_weigths)
		print('avg_AUPR:')
		print(round(np.sum(top_AUPR)/len(top_AUPR),4))
		print('yaronAUPR:')
		print(round(np.sum(yaronAUPR)/len(yaronAUPR),4))
		


	print('\n########################################\n Hyper Params\n########################################\n')
	print('Model number {}'.format(modelNum))
	printDict(paramsDict)


if __name__ == '__main__':
	_main()
