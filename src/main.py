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


def _main(numOfRuns=3):

	ind = 0
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
	runFolderDir = util_functions.startLogging(isDump=False)

	# Permutate paramsDict to
	paramsDict = getParamsDict(modelNum)

	# Run the nn over num of samples
	resDict = {}
	avg_AUPR = np.zeros(numOfRuns)
	top_AUPR = np.zeros(numOfRuns)
	AUPR = np.zeros([numOfRuns,6])
	selex_num = []
	predictions = []
	for sample in range(numOfRuns):
		print('++++++++++++ sample number %i: ++++++++++++' % sample)
		# Get random sample data and export the files as list
		sampleNum, filesList = util_functions.getTrainSampleFromList(dataRoot=os.path.realpath(__file__ + '/../../')+'/train',
		                                                             listOfSamples=listOfSamples,
		                                                             ind=ind)
		ind += 1
		selex_num.append(len(filesList) - 3)
		selex_index = np.linspace(1, selex_num[sample], selex_num[sample])
		selex = 1
		dataPipe = read_data.DataPipeline(filesList, int(selex))
		test_length = len(dataPipe.testData)
		tmp_predictions = np.zeros([selex_num[sample], test_length])
		for selex in selex_index:
			#if np.any(selex == [3,4]):
			#	continue
			selex = int(selex)
			print('+++++++++ selex number %i +++++++++' % selex)
			if selex != 1:
				# Create data pipeline obj
				dataPipe = read_data.DataPipeline(filesList, selex)

			# Create and train the model
			if selex == 1:
				model = SimpleModel(paramsDict, modelNum)
			model.model.summary()
			model.train(tain_generator=dataPipe.train_generator, validation_generator=None,
		            steps_per_epoch=5000, n_epochs=1, n_workers=6)



			# Evaluate the model
			tmp_predictions[selex-1,:] = np.squeeze(model.predict(dataPipe.test_generator, 6))
			#predictions = model.predict(dataPipe.test_generator, 6)
			if selex == selex_num[sample]:
				print('selex %i and final AUPR:' % selex)
			else:
				print('selex %i AUPR:' % selex)
			AUPR[sample,selex-1]=util_functions.getAUPR(dataPipe.testData, tmp_predictions[selex-1,:])

		predictions.append(tmp_predictions)
		weights = np.linspace(1,selex_num[sample]/2,selex_num[sample])
		avg_predictions = np.average(predictions[sample], axis=0,weights=weights)
		top_predictions = (predictions[sample][selex_num[sample]-1,:]*1.8 + predictions[sample][selex_num[sample]-2,:]*1.3 +
						   predictions[sample][selex_num[sample]-3,:]*1)/(1.8+1.3+1)
		print('average AUPR:')
		avg_AUPR[sample] = util_functions.getAUPR(dataPipe.testData, avg_predictions)
		print('top AUPR:')
		top_AUPR[sample] = util_functions.getAUPR(dataPipe.testData, top_predictions)

		#resDict[str(sampleNum)] = round(AUPR[sample,selex_num-1], 5)



	print('\n########################################\n Hyper Params\n########################################\n')
	print('Model number {}'.format(modelNum))

	printDict(paramsDict)
	print('\n########################################\n Results\n########################################\n')
	#for k,v in resDict.items():
		#print('Sample number {} : my AUPR {} : yaron AUPR {}'.format(k, round(v, 4),round(yaronAUPR[listOfSamples.index(int(k)) + 1],4)))
	#	print('Sample number {} : my AUPR {} : yaron AUPR {}'.format(k, round(v, 4),
																#	 round(yaronAUPR[k],
																#		   4)))

	for index,AUPR_ in enumerate(yaronAUPR):
		print('sample number: %s' % listOfSamples[index])
		print('yaronAUPR: %f' % round(AUPR_,4))
		print('MyAUPR: %f' % round(AUPR[index,selex_num[index]-1],4))
	print('My average AUPR %f:' % round(np.sum(AUPR[:,selex_num[index]-1])/len(listOfSamples),4))
	print('Yaron average AUPR %f:' % round(sum(yaronAUPR)/len(yaronAUPR), 4))
	#print('My average AUPR : {}'.format(round(sum(resDict.values())/len(resDict),4)))
	#print('Yarons average AUPR : {}'.format(round(sum(yaronAUPR) / len(yaronAUPR),4)))



if __name__ == '__main__':
	_main()
