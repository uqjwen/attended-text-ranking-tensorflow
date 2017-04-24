import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D


class Model():
	def __init__(self, config):
		self.left = tf.placeholder(tf.float32, [None, config.max_left_len, config.embedding_size])
		self.right  = tf.placeholder(tf.float32, [None, config.max_left_len, config.embedding_size])
		self.label = tf.placeholder(tf.float32, [None,2])

		pool_outputs_left = []
		pool_outputs_right = []

		for i,filter_size in enumerate(config.filter_sizes):
			with tf.name_scope("left-conv-maxpool-{}".format(i)):
				conv1d = Convolution1D(nb_filter = config.nb_filter,
										filter_length = filter_size,
										border_mode = 'valid',
										activation = 'relu',
										input_shape=[config.max_left_len, config.embedding_size])
				left_conv1d = conv1d(self.left)
				left_maxpool = MaxPooling1D(pool_length = config.max_left_len - filter_size+1)(left_conv1d)
				pool_outputs_left.append(left_maxpool)
			with tf.name_scope("right-conv-maxpool-{}".format(i)):
				conv1d = Convolution1D(nb_filter = config.nb_filter,
										filter_length = filter_size,
										border_mode = 'valid',
										activation = 'relu',
										input_shape = [config.max_right_len, config.embedding_size])
				right_conv1d = conv1d(self.right)
				right_maxpool = MaxPooling1D(pool_length=config.max_right_len - filter_size+1)(right_conv1d)
				pool_outputs_right.append(right_maxpool)


		num_filters_total = len(config.filter_sizes)*config.nb_filter

		h_left = tf.reshape(tf.concat(pool_outputs_left, 2), [-1, num_filters_total])
		h_right = tf.reshape(tf.concat(pool_outputs_right,2), [-1, num_filters_total])

		similarity = tf.multiply(h_left, h_right)

		h_concate = tf.concat([h_left, h_right, similarity],1)

		h_layer = Dense(input_dim = 3*num_filters_total, units = config.hidden_size, activation='relu')

		h_hidden = h_layer(h_concate)

		self.logits = Dense(input_dim = config.hidden_size, units = 2)(h_hidden)



		self.predictions = tf.argmax(self.logits,1)



		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.label, logits = self.logits))

		correction_prediction = tf.equal(self.predictions, tf.argmax(self.label,1))
		self.accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))



#config. embedding_size, max_left_len, max_right_len, nb_filter, filter_sizes, hidden_size