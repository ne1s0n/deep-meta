import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import matthews_corrcoef

def extra_metrics(y, yhat):
	"""returns a dictionary with a set of extra statistics on performance"""
	res = {}
	res['val_MCC'] = matthews_corrcoef(y, yhat)

	return(res)

def save_results(conf_dict, metrics_dict, outfile):
	"""Saves the passed configuration and metrics data in the passed outfile"""
	
	#let's have a big dict to save
	d = conf_dict
	d.update(metrics_dict)
	
	#if outfile does not exist there are extra steps required
	if not os.path.isfile(outfile):
		#let's ensure the path is there
		p = os.path.dirname(os.path.abspath(outfile))
		pathlib.Path(p).mkdir(parents=True, exist_ok=True) 
		#let's add at least the header
		with open(outfile, 'w') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=d.keys())
			writer.writeheader()
		
	#at this point the output file exists for sure. Let's append the data
	with open(outfile, 'a+') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=d.keys())
		writer.writerow(d)
	
	#done
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
		
		#rest of the plot
		plt.xlabel('Epochs')
		plt.title(title + ' - ' + metric)
		plt.legend()
		plt.show()

