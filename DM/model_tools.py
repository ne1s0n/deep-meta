from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, Input, BatchNormalization
import keras.metrics

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
