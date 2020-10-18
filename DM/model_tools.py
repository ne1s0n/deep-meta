from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, Input, BatchNormalization
import keras.metrics
import numpy as np

def model_seq2_update(p):
	"""updated the passed parameter dict according to the contained growth
	factor"""
	
	#the number of conv units simply grows
	p['conv_units_internal'] = p['conv_units_internal'] * (p['beta'] ** p['phi']) 
	p['conv_units'] = np.floor(p['conv_units_internal'])
	
	#the size of conv units shrinks, but never goes below 2
	p['conv_size_internal'] = p['conv_size_internal'] / (p['gamma'] ** p['phi']) 
	p['conv_size'] = np.max([2, np.floor(p['conv_size_internal'])])

	
def model_seq2_layer(p):
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
	
def model_seq2(phi, input_shape, output_units, lr):
	"""Implementing a variable size model, linked to a single parameter phi"""
	m = Sequential()
	
	#putting all the parameters in a handy dict
	p = {
		#these are fixed, for now
		'dropout_rate' : 0.2,
		'pool_size' : 2,
		'alpha' : 2,  #base for growing layers
		'beta'  : 2,  #base for growing conv units count
		'gamma' : 2,  #base for shrinking conv units kernels
	
		#these represent the base model
		'layers' : 3,
		'conv_units' : 32,
		'conv_size'  : 32,
		
		#just taking notes
		'phi' : phi
	}
	
	
	#some derived parameters
	p['conv_units_internal'] = p['conv_units']
	p['conv_size_internal'] = p['conv_size']
	p['final_layers'] = np.floor(p['layers'] * (p['alpha'] ** p['phi']))
	
	#building the model
	m = Sequential()
	
	#adding the layers and updating the parameters accordingly to
	#growth factor
	for l in range(p['final_layers']):
		m.add(model_tools.model_seq2_layer(p))
		model_tools.model_seq2_update(p)
	
	#top layers
	m.add(Flatten())
	#binary or multiclass?
	if output_units > 2:
		m.add(Dense(units = output_units, activation='softmax'))
		target_loss = 'categorical_crossentropy'
		target_accuracy = ['categorical_accuracy']
	else:
		m.add(Dense(units = 1, activation='sigmoid'))
		target_loss = 'binary_crossentropy'
		target_accuracy = ['binary_accuracy']

	#the model is declared, but we still need to compile it to actually
	#build all the data structures
	m.compile(optimizer = Adam(learning_rate = lr),
		loss = target_loss, 
		metrics = target_accuracy)
	
	return(m)
	
	


def get_model_seq1(conv_filters, conv_kernel_sizes, dropout_rates, pool_sizes, input_shape, output_units, lr):
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
		target_loss = 'categorical_crossentropy'
		target_accuracy = ['categorical_accuracy']
	else:
		model.add(Dense(units = 1, activation='sigmoid'))
		target_loss = 'binary_crossentropy'
		target_accuracy = ['binary_accuracy']

	#the model is declared, but we still need to compile it to actually
	#build all the data structures
	model.compile(optimizer = Adam(learning_rate = lr),
		loss = target_loss, 
		metrics = target_accuracy)

	return(model) 

def get_model_Yıldırım(input_shape, output_units, lr):
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
		target_loss = 'categorical_crossentropy'
		target_accuracy = ['categorical_accuracy']
	else:
		model.add(Dense(units = 1, activation='sigmoid'))
		target_loss = 'binary_crossentropy'
		target_accuracy = ['binary_accuracy']

	#the model is declared, but we still need to compile it to actually
	#build all the data structures
	model.compile(optimizer = Adam(learning_rate = lr),
		loss = target_loss, 
		metrics = target_accuracy)

	return(model) 
