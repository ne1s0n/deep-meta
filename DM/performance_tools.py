import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import matthews_corrcoef

def extra_metrics(y, yhat):
	res = {}
	res['val_MCC'] = matthews_corrcoef(y, yhat)

	return(res)

def save_performances(conf_dict, metrics_dict, outfile):
	#TODO
	#if outfile is already there work in append,
	#otherwise work in new-file-creation
	#the new data (use last_metrics) to the list of dicts
	#save
	return(None)

def last_metrics(h, n=5):
	"""returns the average (over the last n epochs) of metrics and losses 
	from a history log as returned by a keras model .fit() method"""
	res = {}
	for metric in h.history.keys():
		res[metric] = np.mean(h.history[metric][-n:])

	return(res)

def plot_history(h, title):
	"""Plots loss and all available metrics starting from a history log as returned 
	by a keras model .fit() method"""
	for metric in h.history.keys():
		#ignoring metrics on validation set, which are implied when
		#plotting on training set
		if metric.startswith('val_'):
			continue

		#if we get here we found a metric on the training set,
		#let's plot it
		plt.plot(h.history[metric], label = "Train set")

		#is it present in the val set, too?
		k = "val_" + metric
		if k in h.history:
			plt.plot(h.history[k], label = "Validation set")
			plt.xlabel('Epochs')
			plt.title(title + ' - ' + metric)
			plt.legend()
			plt.show()

