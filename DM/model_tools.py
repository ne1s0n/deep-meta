from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, Input
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
