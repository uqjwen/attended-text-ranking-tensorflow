import sys
import os
import time
import datetime
import itertools
import numpy as np
import tensorflow as tf
from collections import Counter
from data_loader import Data_Loader
# from model import Model
from model_v1 import Model
# from Ranking import Ranking


# config. embedding_size, max_left_len, max_right_len, nb_filter, filter_sizes, hidden_size

tf.flags.DEFINE_integer("embedding_size", 100, "Dimensionality of character embedding (default: 100)")
tf.flags.DEFINE_integer("filter_sizes", (2,3), "Comma-separated filter sizes (default: '2,3')")
tf.flags.DEFINE_integer("nb_filter", 64, "Number of filters per filter size (default: 64)")
tf.flags.DEFINE_integer("hidden_size", 100, "Number of hidden layer units (default: 100)")



tf.flags.DEFINE_integer("max_left_len", 56, "max document length of left input")
tf.flags.DEFINE_integer("max_right_len", 56, "max document length of right input")

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")


config = tf.flags.FLAGS


def main():
	# model = Model(config)
	data_loader = Data_Loader(config.batch_size)
	config.max_left_len = data_loader.max_left_len
	config.max_right_len = data_loader.max_right_len

	model = Model(config)
	with tf.Session() as sess:

		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(1e-3)
		grads_and_vars = optimizer.compute_gradients(model.loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
		saver = tf.train.Saver(tf.all_variables(), max_to_keep=20)
		sess.run(tf.initialize_all_variables())


		checkpoint_dir = './checkpoint/'
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print (" [*] Loading parameters success...")
		else:
			print (" [!] Loading parameters failed!!!")

		for e in range(config.num_epochs):
			data_loader.reset_pointer()
			total_batch = int(data_loader.train_size/config.batch_size)
			for b in range(total_batch):
				left,right,label = data_loader.next_batch()
				feed = {model.left: left, model.right:right, model.label:label}
				_,loss,acc = sess.run([train_op, model.loss, model.accuracy], feed_dict = feed)
				sys.stdout.write("\repoch:{}/{},batch:{}/{}, loss:{}, acc:{}".\
					format(e,config.num_epochs, b, total_batch, loss, acc))
				sys.stdout.flush()

				if((e*total_batch+b)%config.evaluate_every==0):
					left,right,label = data_loader.test_data()
					feed = {model.left:left, model.right:right, model.label:label}
					loss,acc = sess.run([model.loss, model.accuracy], feed_dict = feed)
					print("\nvalidation loss: {}, accuracy:{}".format(loss, acc))
				if((e*total_batch+b)%config.checkpoint_every==0 or\
					(e==config.num_epochs-1 and b == total_batch-1)):
					saver.save(sess, checkpoint_dir+'model.ckpt', global_step = e*total_batch+b)

		


if __name__ == "__main__":
	main()