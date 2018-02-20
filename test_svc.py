import os
import sys

import numpy as np
import sklearn

print(dir(sklearn))
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from classify import data
from classify import svc

from time import time 

if __name__ == '__main__':
	if len(sys.argv) == 1:
		print('Usage: python', sys.argv[0], 'path-to-dir [out dir]')
	else:
		outdir = ''
		if len(sys.argv) == 3:
			outdir = sys.argv[2]
			if outdir[-1] != '/':
				outdir += '/'
			if not os.path.exists(outdir):
				os.makedirs(outdir)

		path = sys.argv[1]

		if not os.path.isdir(path):
			print("Given path is not a directory!")
			sys.exit()

		files = []
		for f in os.listdir(path):
			if os.path.isfile(os.path.join(path, f)) and f.lower().endswith('.json'):
				print("Found file:", f)
				files.append(os.path.join(path, f))

		X = np.empty(shape=(0, 15))
		y = np.empty(shape=(0))
		for file in files:
			sample_data, label = data.process_data(data.read_data_file(file))
			if not label == -1:
				X = np.append(X, [data.create_sample(sample_data)], axis=0)
				y = np.append(y, label)

		print(X.shape)
		print(y.shape)

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

		classifier = svc.create_classifier(X_train, y_train)

		print("Running test:")
		t0 = time()
		y_pred = classifier.predict(X_test)
		print("Done in %0.3fs" % (time() - t0))

		print(classification_report(y_test, y_pred))