
import os
import re
import pandas as pd
import numpy as np

# csvPath = ''
# debug = '/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/runData/RunFolder_24_04_18__10_49_iter_num_22/logFile_10_49__24_04_18.log'
# runDataRootDir = '/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/runData'

runDataRootDir = '/Users/royhirsch/Documents/GitHub/dataBioLog/runData'
csvPath = runDataRootDir + '/summery.csv'

# get all permutaion params as dicts:
logsDicts = []
for root, dirs, files in os.walk(runDataRootDir):
	for fileName in files:
		match = re.search(r'logFile_', fileName)
		if match:
			file = open(os.path.join(root, fileName),'r')
			# file = open(debug, 'r')
			logText = file.read()
			file.close()

			dictParams = dict()
			dictParams['depth'] = (re.findall('depth : ([\w\d\[\], ]*)', logText)[0])
			dictParams['dropout'] = float(re.findall('dropout : ([\d.]*)', logText)[0])
			dictParams['hidden_size'] = int(re.findall('hidden_size : ([123456790.]*)', logText)[0])
			dictParams['lr_decay'] = float(re.findall('lr_decay : ([e123456790.-]*)', logText)[0])
			dictParams['max_pool'] = (re.findall('max_pool : ([123456790]*)', logText)[0])
			dictParams['optimizer'] = re.findall('optimizer : ([\S]*)', logText)[0]

			resList = re.findall('AUPR = ([\d.]*)', logText)
			resList = [float(item) for item in resList]
			dictParams['samples_num'] = len(resList)
			dictParams['mean_AUPR'] = round(np.mean(resList),4)
			dictParams['std_AUPR'] = round(np.std(resList),4)

			logsDicts.append(dictParams)

table = pd.DataFrame(logsDicts ,columns=dictParams.keys())
for dict in logsDicts:
	newRow = pd.DataFrame([dict], columns=dict.keys())
	table = pd.concat([table, newRow], axis=0, ignore_index=True)

table.to_csv(csvPath)

#
# sampleText = '2018-05-02 08:47:42,154 - INFO - Trainer : ++++++ Validation for step num 1000 ++++++\n2018-05-02 08:47:42,155 - INFO - Trainer : Training Accuracy : 0.9795\n2018-05-02 08:47:42,155 - INFO - Trainer : Dice score: 0.1710\n'
# valText = re.findall('(Validation for step num )([0-9])(.*)([\n])(.*)(Training Accuracy : )(.*\d)([\n])(.*)(Dice score: )(.*\d)', sampleText)
