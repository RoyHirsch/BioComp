import re
import numpy as np
import os

root = os.path.realpath(__file__ + "/../")

res = np.zeros([1,130])
for root, dirs, files in os.walk(root):
	for fileName in files:
		match = re.search(r'train[\d.]*o', fileName)
		if match:
			file = open(os.path.join(root, fileName),'r')
			logText = file.read()
			file.close()
			score = float(re.findall('AUPR = ([\d.]*)', logText)[0])
			sample = int(re.findall(r'train([\d].*)o', fileName)[0][:-1])
			res[0][sample] = score

for i in range(len(res)):
	if res[0][i]:
		print('Sample number {} : AUPR : '.format(i, res[0][i]))