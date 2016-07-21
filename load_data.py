import numpy as np

from find_contour import *

# Load data from file
def loadTriangleTrain(file='train.csv'):
	data = np.genfromtxt(file, delimiter=',')

	X = data[:, :-1]
	y = data[:, -1]
	
	return (X, y)
	
	
from pybrain.datasets import ClassificationDataSet

def processData(url, input=100, output=1, nb_classes=2):
	'''
	url: location to image folder
	input: number of nodes in input layer
	output: number of nodes in output layer
	nb_classes: number of classes to classify
	'''
	
	data = ClassificationDataSet(input, output, nb_classes=nb_classes)

	X, y = find_contour(url=url, features=input)
	
	for i in range(X.shape[0]):
		data.addSample(X[i, :] / 256, y[i])

	# convert target into binary
	# (0, 1) = triangle, (1, 0) = other
	data._convertToOneOfMany()
	
	return data