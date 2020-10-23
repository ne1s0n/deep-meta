import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, roc_auc_score

def extra_metrics(y, yhat_scores, binary):
	"""returns a dictionary with a set of extra statistics on performance"""
	res = {}
	if binary:
		res['MCC'] = matthews_corrcoef(y, np.rint(yhat_scores))
		(res['pre'], res['rec'], res['fbeta'], res['supp']) = precision_recall_fscore_support(
			y_true = y, y_pred = np.rint(yhat_scores), pos_label=1, average='binary'
		)
		res['AUC'] = roc_auc_score(y_true = y, y_score = np.rint(yhat_scores))
	else:
		res['MCC'] = matthews_corrcoef(y, yhat_scores.argmax(axis=-1))
		#added to have the same columns in the resulting performance table
		res['pre'] = None
		res['rec'] = None
		res['fbeta'] = None
		res['supp'] = None
		res['AUC'] = None
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

def best_metrics(h):
	"""Returns the best results obtained over all training for the metric
	present in the history log object as returned by a keras model .fit() method. 
	"Best" is defined as minimum for loss, maximum for everything else."""
	res = {}
	for metric in h.history.keys():
		if metric.find('loss') != -1:
			#loss
			res[metric + '_best'] = np.min(h.history[metric])
		else:
			#non loss, things like accuracy and AUC
			res[metric + '_best'] = np.max(h.history[metric])
	return(res)

def last_metrics(h, n=5):
	"""Returns the average (over the last n epochs) of metrics and losses 
	from a history log as returned by a keras model .fit() method"""
	res = {}
	for metric in h.history.keys():
		res[metric] = np.mean(h.history[metric][-n:])

	return(res)

def plot_history(h, title = None, outfile = None, show = False, lines = None, ylim=(-0.05, 1.05)):
	"""Plots either the selected lines or everything available, 
	data is from a history log h as returned by a keras model .fit() method"""
	
	#if the user did not specify a subset we take everything
	if lines is None:
		lines = h.history.keys()
		
	#starting a new plot
	plt.figure() 
	
	#adding each line
	for l in lines:
		plt.plot(h.history[l], label = l)
	
	#rest of the plot
	plt.xlabel('Epochs')
	plt.title(title)
	plt.legend()
	plt.ylim(ylim)
	
	#saving
	if outfile is not None:
		plt.savefig(outfile)
	
	#showing
	if show:
		plt.show()

def format_timedelta(delta, include_seconds = False):
	"""Transforms a datetime.timedelta into a printable string, optionally
	including seconds"""
	hours, remainder = divmod(delta.seconds, 3600)
	minutes, seconds = divmod(remainder, 60)
		
	res = '{:02}:{:02}'.format(int(hours), int(minutes))
	if include_seconds:
		res += ':{:02}'.format(int(seconds))
		
	return(res)
