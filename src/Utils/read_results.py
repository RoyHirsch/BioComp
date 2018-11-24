import re
import os
import numpy as np

# copy from remote dir to local dir:
# this call will filter and copy only the output files from the remote to local dir
#run on local: scp -r colfax:path/to/remote/train/dir/train.o* path/to/local/dir

# script to extract aupr data from local output files
dir = '/Users/royhirsch/Documents/Study/SemB2018/Bioinformatics/res/'
auprd = {}
for _, _, files in os.walk(dir):
	for name in files:
		path = os.path.join(dir, name)
		file = open(path,'r')
		text = file.read()
		file.close()
		num = str(re.findall('train(\d+)',name)[0])

		if len(re.findall('AUPR = (\d.+)',text)) > 0:
			auprd[num] = round(float(re.findall('AUPR = (\d.+)',text)[0]), 5)

listv = [float(x) for x in auprd.values()]
meanval = np.mean(listv)
stdval = np.std(listv)

print('len: ' + str(len(listv)))
print('mean:' + str(meanval))
print('std: ' + str(stdval))
print('values:' + str(listv))