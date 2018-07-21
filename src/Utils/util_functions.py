import os
import logging
import time
import sys
import numpy as np
from sklearn.metrics import average_precision_score


def sortPBM(originalPBMdata, predictionsPerString):

	# Concatenate the original strings and the predicted values:
	# string1 prediction1
	# string2 prediction2
	# string3 prediction3

	con = np.concatenate([originalPBMdata, predictionsPerString], axis=1)
	sortedCon = con[con[:, 1].argsort()][::-1]
	return sortedCon[:][0]

def dumpPDMtoFile(folder, fileName, sortedPDMStrings):
	fullFilePath = folder + fileName + '.txt'
	np.savetxt(fullFilePath, sortedPDMStrings, delimiter=" ", fmt="%s")
	return

'''
How to use precision_recall_curve:
>>> y_true = np.array([0, 0, 1, 1])
>>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
>>> precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
'''
def getAUPR(groundTrue, predict):

	# Generate the ground true label
	true = [int(x) for x in np.append(np.ones(100), np.zeros(len(groundTrue) - 100), axis=0)]

	# The area under the precision-recall curve is AUPR
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
	# Output: runFolderDir - the new folder dir if isDump is True, else None
	#########################################################################
def startLogging(isDump):

	# Init a logger set logging level
	log = logging.getLogger(__name__)
	logFormat = '%(asctime)s - %(levelname)s - %(module)s : %(message)s'
	logging.basicConfig(format=logFormat, stream=sys.stdout, level=logging.INFO)
	logLevel = logging.INFO

	logStr = time.strftime('logFile_%H_%M_%S__%d_%m_%y') + '.log'

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

class StreamToLogger(object):
	# taken from:
	# https://www.electricmonk.nl/log/2011/08/14/redirect-stdout-and-stderr-to-a-logger-in-python/

	def __init__(self, logger, log_level=logging.INFO):
		self.logger = logger
		self.log_level = log_level
		self.linebuf = ''

	def write(self, buf):
		for line in buf.rstrip().splitlines():
			self.logger.log(self.log_level, line.rstrip())

	def flush(self):
		pass

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


def getTrainSampleFromList(dataRoot, ind):
	listOfSamples = [57, 59, 86, 80, 41, 54, 9, 108, 73, 30, 62, 96, 5, 50, 74, 24, 99, 101, 68, 31]
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

def getYaronAuprPerSample(ind):
	yaronAUPR = [0.594610698,0.514773825,0.4717869,0.341958113,0.250109638,0.246641534,0.239458146,0.237103258,0.221448961,0.173986496,0.172595686,0.167862911,0.114530832,0.113215591,0.103344651,0.002081527,0.002065405,0.002049701,0.087953909,0.037662088]
	return yaronAUPR[yaronAUPR.index(ind)]