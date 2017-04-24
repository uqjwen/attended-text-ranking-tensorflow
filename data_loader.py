import numpy as np 
import nltk
from keras.utils import np_utils

file = ['./data/train.txt', './data/text.txt']
glove_file = './glove.6B.100d.txt'
class Data_Loader():
	def __init__(self, batch_size):
		self.embedding_index = {}
		self.batch_size = batch_size
		self.embedding_size = int(glove_file.split('.')[3][:-1])

		fr = open(glove_file)
		for line in fr:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype = 'float32')
			self.embedding_index[word] = coefs
		fr.close()
		self.process_data()


	def load_data(self, filename):
		fr = open(file[0])
		data = fr.readlines()
		fr.close()
		text = [line.strip().split(',') for line in data]
		left,right,label = zip(*text)
		def word_tokenize(tokens):
			return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

		left = list(map(word_tokenize, left))
		right = list(map(word_tokenize, right))
		label = np_utils.to_categorical(label,2).astype(float)
		return left,right,label

	def process_data(self):
		self.train_left, self.train_right, self.train_label = self.load_data(file[0])
		self.test_left, self.test_right, self.test_label = self.load_data(file[1])
		self.train_size = len(self.train_left)
		self.test_size = len(self.test_left)

		max_train_left = max(list(map(len, self.train_left)))
		max_train_right = max(list(map(len, self.train_right)))
		max_test_left = max(list(map(len, self.test_left)))
		max_test_right = max(list(map(len, self.test_right)))

		self.max_left_len = max(max_train_left, max_test_left)
		self.max_right_len = max(max_train_right, max_test_right)



	def reset_pointer(self):
		self.pointer = 0

	def pad_transform_sent(self, maxlen, sent):
		ret = []
		need_pad = maxlen-len(sent)
		for i in range(need_pad):
			ret.append(np.random.uniform(-1,1,self.embedding_size))
		for word in sent:
			ret.append(self.embedding_index[word] if word in self.embedding_index else np.random.uniform(-1,1,self.embedding_size))
		return np.array(ret)


	def pad_transform_sents(self, maxlen, sents):
		ret = []
		# ret.append(self.pad_transform_sent(maxlen, sent) for sent in sents)
		for sent in sents:
			ret.append(self.pad_transform_sent(maxlen, sent))
		return np.array(ret)





	def next_batch(self):
		begin = self.pointer*self.batch_size
		end = (self.pointer+1)*self.batch_size
		self.pointer += 1


		batch_right = self.pad_transform_sents(self.max_right_len, self.train_right[begin:end])
		batch_left = self.pad_transform_sents(self.max_left_len, self.train_left[begin:end])
		batch_label = self.train_label[begin:end]

		return batch_left, batch_right, batch_label

	def test_data(self):
		left = self.pad_transform_sents(self.max_left_len, self.test_left)
		right = self.pad_transform_sents(self.max_right_len, self.test_right)
		label = self.test_label

		return left,right,label

if __name__ == "__main__":
	data_loader = Data_Loader(batch_size = 64)
	data_loader.reset_pointer()
	data,_,_ = data_loader.next_batch()