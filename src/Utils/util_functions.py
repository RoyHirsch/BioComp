import numpy as np
import os
import logging
import time
import sys
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


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

def getAUPR(groundTrue, prediction, isPrint):

	# Generate the ground true label
	label = np.zeros([len(groundTrue)])
	label[:100] = 1

	precision, recall, _ = precision_recall_curve(label, prediction)

	aupr = metrics.auc(recall, precision)
	print('AUPR={0:0.2f}',format(aupr))

	if isPrint:
		import matplotlib.pyplot as plt
		plt.step(recall, precision, color='b', alpha=0.2,where='post')
		plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')

		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.0])
		plt.title('2-class Precision-Recall curve')

	return

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
def startLogging(isDump=False):

	# Init a logger set logging level
	logFormat = '%(asctime)s - %(levelname)s - %(module)s : %(message)s'
	logging.basicConfig(format=logFormat, stream=sys.stdout, level=logging.INFO)
	logLevel = logging.INFO

	logStr = time.strftime('logFile_%H_%M__%d_%m_%y') + '.log'

	if isDump:
		createFolder(os.path.realpath(__file__ + "/../../../"), 'runData')
		runFolderStr = time.strftime('RunFolder_%H_%M__%d_%m_%y')
		createFolder(os.path.realpath(__file__ + "/../../../") + "/runData/", runFolderStr)
		runFolderDir = os.path.realpath(__file__ + "/../../../") + "/runData/" + runFolderStr

		fileHandler = logging.FileHandler(runFolderDir+'/'+logStr)
		fileHandler.setFormatter(logging.Formatter(logFormat))
		fileHandler.setLevel(logLevel)
		logging.getLogger('').addHandler(fileHandler)

		logging.info("Logger was created, isDump is True. Log is in {}.".format(runFolderDir))
		return runFolderDir

	logging.info("Logger was created, isDump is False.")
	return None
