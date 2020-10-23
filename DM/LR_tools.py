import os
import numpy as np
from pyimagesearch_lrfinder import learningratefinder


def bestLR(model, trainData, startLR = 1e-10, endLR = 1e+1,
	batchSize=32, class_weight=None, epochs=5, sampleSize=2048,
	verbose=1, outfile=None):
	"""Returns the best LR based on the test from learningratefinder"""

	#save the model weights
	we = model.get_weights()

	#use pyImageSearch LR finder
	lrf = learningratefinder.LearningRateFinder(model)
	lrf.find(
		trainData = trainData, 
		startLR = startLR, endLR = endLR,
		batchSize = batchSize,
		class_weight = class_weight,
		epochs = epochs, 
		sampleSize = sampleSize,
		verbose = verbose,
		outfile=None)

	#restore initial weights
	model.set_weights(we)
	
	#extracting and returning the best LR
	i = np.argmin(lrf.losses)
	return(lrf.lrs[i])
