import os
import logging
import time
import sys
import numpy as np
from sklearn.metrics import average_precision_score
from keras import backend as K
import json

#########################################################################
# Description: Reads the model parameters from external json file into a dict.
#
# Input: config_path
# Output: config
#########################################################################
def get_model_parameters(config_path):
    try:
        with open(config_path) as config_json:
            config = json.load(config_json)
            config_json.close()
        return config
    except NameError as ex:
        print("Read Error: no file named %s" % config_path)
        raise ex

#########################################################################
# Description: Concatenate the original strings and the predicted values:
#              string1 prediction1
#              string2 prediction2
#              string3 prediction3
#              Sorts the string by the prediction values and extracts
#              numpy array of the sorted string.
#
# Input: originalPBMdata , predictionsPerString
# Output: sortemPBMdata
#########################################################################
def sortPBM(originalPBMdata, predictionsPerString):
	originalPBMdata = np.array(originalPBMdata).reshape(-1, 1)
	con = np.concatenate([originalPBMdata, predictionsPerString], axis=1)
	sortedCon = con[con[:, 1].argsort()][::-1]
	return sortedCon[:, 0]

#########################################################################
# Description: Extracts the sorted PBM numpy array into a .txt file
# Input: folder, fileName, sortedPDMStrings
# Output: creates a .txt file
#########################################################################
def dumpPBMtoFile(folder, fileName, sortedPDMStrings):
	fullFilePath = os.path.join(folder, fileName + '.txt')
	np.savetxt(fullFilePath, sortedPDMStrings, delimiter=" ", fmt="%s")
	print('The sorted PBM file was saved to {}'.format(fullFilePath))
	return

#########################################################################
# Description: Calculates the AUPR of the model, uses the ground true
#              labeling and the model's predictions.
#
# Input: groundTrue, predict
# Output: aupr
#########################################################################
def getAUPR(groundTrue, predict):
	# Generate the ground true label, the top 100 strings will be positive.
	true = [int(x) for x in np.append(np.ones(100), np.zeros(len(groundTrue) - 100), axis=0)]

	# The area under the precision-recall curve is the AUPR
	aupr = average_precision_score(true, predict)
	print('AUPR = {}'.format(np.round(aupr, 4)))
	return aupr


#####################################################################################
# Operation functions
#####################################################################################

def createFolder(homeDir, folderName):
	directory = homeDir + '/' + folderName
	if not os.path.exists(directory):
		os.makedirs(directory)

#########################################################################
# Description: Generate a logger format for better documentation.
# Input: isDump - if True is creates a folder and saves the log in it.
# 	              if False, the log will only be displayed at the stdout.
#                 The logger also records the stdout in real-time.
#
# Output: runFolderDir - the new folder dir if isDump is True, else None
#########################################################################
def startLogging(isDump):

	# Init a logger set logging level
	log = logging.getLogger(__name__)
	logFormat = '%(asctime)s - %(levelname)s - %(module)s : %(message)s'
	logging.basicConfig(format=logFormat, stream=sys.stdout, level=logging.INFO)
	logLevel = logging.INFO

	logStr = time.strftime('logFile_%H_%M_%S__%d_%m_%y') + '.log'

	# If true, create a folder and a log text file.
	if isDump:
		createFolder(os.path.realpath(__file__ + "/../../../"), 'runData')
		runFolderStr = time.strftime('RunFolder_%H_%M_%S__%d_%m_%y')
		createFolder(os.path.realpath(__file__ + "/../../../") + "/runData/", runFolderStr)
		runFolderDir = os.path.realpath(__file__ + "/../../../") + "/runData/" + runFolderStr

		fileHandler = logging.FileHandler(runFolderDir+'/'+logStr)
		fileHandler.setFormatter(logging.Formatter(logFormat))
		fileHandler.setLevel(logLevel)

		logging.getLogger('').addHandler(fileHandler)

		# Added an option to log stdout in realtime
		sl = StreamToLogger(logging.getLogger(''), logging.INFO)
		sys.stdout = sl

		# Added an option to log stderr in realtime
		sl = StreamToLogger(logging.getLogger(''), logging.ERROR)
		sys.stderr = sl

		logging.info("Logger was created, isDump is True. Log is in {}.".format(runFolderDir))
		return runFolderDir

	logging.info("Logger was created, isDump is False.")
	return None

# Helper class to record the stdout of the run in the logger.
class StreamToLogger(object):

	def __init__(self, logger, log_level=logging.INFO):
		self.logger = logger
		self.log_level = log_level
		self.linebuf = ''

	def write(self, buf):
		for line in buf.rstrip().splitlines():
			self.logger.log(self.log_level, line.rstrip())

	def flush(self):
		pass

#########################################################################
# Description: Returns a sample experiment selected randomly.
#              This function was used during the development of the model.
# Input: dataRoot
# Output: sampleNum, filesList
#########################################################################
def getTrainSample(dataRoot):
	sampleNum = int(np.random.randint(1,123,1))

	fileList = os.listdir(dataRoot)
	filesList = ['']

	pbmFilename = 'TF{}_pbm.txt'.format(sampleNum)
	filesList.append(pbmFilename)

	for i in range(10):
		selexFilename = 'TF{}_selex_{}.txt'.format(sampleNum, i)
		if selexFilename in fileList:
			filesList.append(selexFilename)
		else:
			break
	return sampleNum, filesList

#########################################################################
# Description: Returns a sample experiment from a pre-defined list.
#              This function was used during the development of the model.
# Input: dataRoot
# Output: sampleNum, filesList
#########################################################################
def getTrainSampleFromList(dataRoot, listOfSamples, ind):

	#listOfSamples = [57, 59, 86, 80, 41, 54, 9, 108, 73, 30, 62, 96, 5, 50, 74, 24, 99, 101, 68, 31]
	sampleNum = listOfSamples[ind]

	fileList = os.listdir(dataRoot)
	filesList = ['']

	pbmFilename = 'TF{}_pbm.txt'.format(sampleNum)
	filesList.append(pbmFilename)

	for i in range(10):
		selexFilename = 'TF{}_selex_{}.txt'.format(sampleNum, i)
		if selexFilename in fileList:
			filesList.append(selexFilename)
		else:
			break
	return sampleNum, filesList
