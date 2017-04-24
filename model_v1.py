import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D
from tensorflow.contrib import rnn

def lstm_layer(inputs, time_steps, rnn_size, name):
	with tf.variable_scope(name):

		inputs = tf.split(inputs, time_steps, 1)
		inputs = [tf.squeeze(input_,[1]) for input_ in inputs]
		lstm = rnn.BasicLSTMCell(rnn_size, state_is_tuple = True)
		outputs, _states = rnn.static_rnn(lstm, inputs, dtype = tf.float32)
		outputs = [tf.expand_dims(output, 1) for output in outputs]
		outputs = tf.concat(outputs,1)
		return outputs


def att_layer(inputs, time_steps, rnn_size, name):
	with tf.variable_scope(name):
		w = tf.get_variable('w',
							shape=[rnn_size,1],
							initializer = tf.contrib.layers.xavier_initializer())
		b = tf.Variable(tf.constant(0.1, shape=[1]))
		c_output = tf.matmul(tf.reshape(inputs, [-1, rnn_size]), w)+b
		n_output = tf.expand_dims(tf.nn.softmax(tf.reshape(c_output, [-1,time_steps])),-1)
		# n_output = tf.expand_dims(tf.nn.softmax(tf.squeeze(c_output, -1)),-1)
		f_output = tf.reduce_sum(tf.multiply(inputs, n_output),1)
		return f_output

	# with tf.variable_scope(name):
	# 	w = tf.get_variable('w',
	# 						shape=[time_steps, rnn_size],
	# 						initializer = tf.contrib.layers.xavier_initializer())
	# 	b = tf.Variable(tf.constant(0.1, shape=[rnn_size]))
	# 	c_output = tf.multiply(inputs, w)+b
	# 	n_output = tf.expand_dims(tf.nn.softmax(tf.reduce_sum(c_output,-1)),-1)
	# 	f_output = tf.reduce_sum(tf.multiply(inputs, n_output), 1)
	# 	return f_output

	# inputs = tf.split(inputs, time_steps, 1)
	# return tf.squeeze(inputs[-1], [1])

	# return tf.reduce_mean(inputs, 1)








class Model():
	def __init__(self, config):
		self.left = tf.placeholder(tf.float32, [None, config.max_left_len, config.embedding_size])
		self.right  = tf.placeholder(tf.float32, [None, config.max_left_len, config.embedding_size])
		self.label = tf.placeholder(tf.float32, [None,2])

		lstm_left = lstm_layer(self.left, config.max_left_len, config.embedding_size, 'lstm_left')
		lstm_right = lstm_layer(self.right, config.max_right_len, config.embedding_size, 'lstm_right')

		att_left = att_layer(lstm_left, config.max_left_len, config.embedding_size, 'att_left')
		att_right = att_layer(lstm_right, config.max_right_len, config.embedding_size, 'att_right')
		h_concat = tf.concat([att_left, att_right, tf.multiply(att_left, att_right)], 1)

		# lstm_last_left = lstm_left[-1]
		# lstm_last_right = lstm_right[-1]
		# h_concat = tf.concat([lstm_last_left, lstm_last_right, tf.multiply(lstm_last_left, lstm_last_right)],1)

		self.logits = Dense(input_dim = config.embedding_size*3, units = 2)(h_concat)






		self.predictions = tf.argmax(self.logits,1)



		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.label, logits = self.logits))

		correction_prediction = tf.equal(self.predictions, tf.argmax(self.label,1))
		self.accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))



#config. embedding_size, max_left_len, max_right_len, nb_filter, filter_sizes, hidden_size