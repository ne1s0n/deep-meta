from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, Input, BatchNormalization
import keras.metrics
import numpy as np
from DM import performance_tools, model_tools

	
def _model_seq2_layer(p):
	"""Returns a Sequential model representing a layer in seq2 model. It
	contains Conv1D, Conv1D, BatchNormalization, MaxPooling1D. The 
	convolutions are identical (same filter size, same number of units). 
	Strides are fixed: 1 for convs, pool_size for pools. Padding is always
	'same'. The argument p is a dict with expected keys: 
	- conv_units
	- conv_size
	- dropout_rate
	- pool_size
	"""
	
	m = Sequential()
	m.add(Conv1D(filters = p['conv_units'], activation='relu', kernel_size = p['conv_size'], strides = 1, padding = 'same'))
	m.add(Conv1D(filters = p['conv_units'], activation='relu', kernel_size = p['conv_size'], strides = 1, padding = 'same'))
	m.add(BatchNormalization())	
	m.add(Dropout(rate=p['dropout_rate']))
	m.add(MaxPooling1D(pool_size=p['pool_size'], padding='same'))

	return(m)
	
def model_seq2(theta, input_shape, output_units, dropout_rate=0.2, pool_size=2, base_layers=4, base_conv_units=32, base_conv_size=32):
	"""Implementing a variable size model, linked to a single parameter: theta"""
	
	#params to be used for building the net
	p = {
		'dropout_rate' : dropout_rate,
		'pool_size' : pool_size,
		'base_layers' : base_layers,
		'base_conv_units' : base_conv_units,
		'base_conv_size'  : base_conv_size
		}
	
	#computing phase transition points
	layers = int(base_layers * theta)
	phase_length = layers / 4.0
	limit1 = int(np.round(phase_length))
	limit2 = int(np.round(2 * phase_length))
	limit3 = int(np.round(3 * phase_length))

	#building the phase plan
	layer_phase = np.zeros(layers)
	layer_phase[0:limit1] = 0
	layer_phase[limit1:limit2] = 1
	layer_phase[limit2:limit3] = 2
	layer_phase[limit3:] = 3
	
	#building the model
	m = Sequential()
	
	#let's start with the input layer
	m.add(Input(shape = input_shape))

	for i in range(layers):
		factor = 2 ** layer_phase[i]
		p['conv_units'] = int(p['base_conv_units'] * factor)
		p['conv_size']  = int(p['base_conv_size']  / factor)
		m.add(model_tools._model_seq2_layer(p))
		
	#top layers
	m.add(Flatten())
	
	#binary or multiclass?
	if output_units > 2:
		m.add(Dense(units = output_units, activation='softmax'))
	else:
		m.add(Dense(units = 1, activation='sigmoid'))

	return(m)

def get_model_seq1(conv_filters, conv_kernel_sizes, dropout_rates, pool_sizes, input_shape, output_units):
	#declare a base model
	model = Sequential()

	#let's start with the input layer
	model.add(Input(shape = input_shape))

	#for each specified layer
	for i in range(len(conv_kernel_sizes)):
		#adding stuff, but only if not None
		if (conv_filters[i] is not None) and (conv_kernel_sizes[i] is not None):
			model.add(Conv1D(filters = conv_filters[i], activation='relu', 
			kernel_size = conv_kernel_sizes[i], strides = 1, padding = 'same'))
			model.add(BatchNormalization())	

		if dropout_rates[i] is not None:   
			model.add(Dropout(rate=dropout_rates[i]))
		if pool_sizes[i] is not None:
			model.add(MaxPooling1D(pool_size=pool_sizes[i], padding='same'))

	#top layers
	model.add(Flatten())
	#binary or multiclass?
	if output_units > 2:
		model.add(Dense(units = output_units, activation='softmax'))
	else:
		model.add(Dense(units = 1, activation='sigmoid'))

	return(model) 

def get_model_Yıldırım(input_shape, output_units):
	"""Model from:
		Yıldırım, Özal, et al. 
		"Arrhythmia detection using deep convolutional neural network with long duration ECG signals." 
		Computers in biology and medicine 102 (2018): 411-420.
	"""
	#declare a base model
	model = Sequential()

	#let's start with the input layer
	model.add(Input(shape = input_shape))
	
	#adding the computing layers
	model.add(Conv1D(filters = 128, activation='relu', kernel_size = 50, strides = 3, padding = 'same'))
	model.add(BatchNormalization())	
	model.add(MaxPooling1D(pool_size=2, strides=3, padding='same'))
	model.add(Conv1D(filters = 32, activation='relu', kernel_size = 7, strides = 1, padding = 'same'))
	model.add(BatchNormalization())	
	model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
	model.add(Conv1D(filters = 32, activation='relu', kernel_size = 10, strides = 1, padding = 'same'))
	model.add(Conv1D(filters = 128, activation='relu', kernel_size = 5, strides = 2, padding = 'same'))
	model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
	model.add(Conv1D(filters = 256, activation='relu', kernel_size = 15, strides = 1, padding = 'same'))
	model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
	model.add(Conv1D(filters = 512, activation='relu', kernel_size = 5, strides = 1, padding = 'same'))
	model.add(Conv1D(filters = 128, activation='relu', kernel_size = 3, strides = 1, padding = 'same'))
	
	#top layers
	model.add(Flatten())
	model.add(Dense(units = 512, activation='relu'))
	model.add(Dropout(rate = 0.1))
	
	#binary or multiclass?
	if output_units > 2:
		model.add(Dense(units = output_units, activation='softmax'))
	else:
		model.add(Dense(units = 1, activation='sigmoid'))

	return(model) 
