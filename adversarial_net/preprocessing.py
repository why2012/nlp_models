from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import pandas as pd
import numpy as np
import nltk
import pickle
from abc import ABCMeta, abstractmethod
from collections import defaultdict, namedtuple
from functools import reduce
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
import re
import copy
from adversarial_net.utils import MutexVariable

def auto_padding(X_list, sentence_size = 100, unkown_word_indicator = 0):
	t = [len(x) for X in X_list for x in X]
	min_len = min(t)
	max_len = max(t)
	return_X_list = []
	for X in X_list:
		return_X_list.append(np.zeros((X.shape[0], sentence_size), dtype=np.int32))
		for i, x in enumerate(X):
			if len(x) < sentence_size:
				padding_size = sentence_size - len(x)
				x = np.pad(x, [(0, padding_size)], "constant", constant_values=[unkown_word_indicator] * 2)
			return_X_list[-1][i][:] = x[:sentence_size]
	return return_X_list, (min_len, max_len)

# like conv layer with strides and padding
# if padding == valid: output_size = ceil(input_size - n_gram + 1) / strides
# if padding == same: output_size = ceil(input_size / strides)
# even padding size on left, odd padding size on right
def make_n_gram(words, n_gram_value, strides = 1, padding = "valid"):
	padding = padding.lower()
	if padding == "valid":
		pass
	elif padding == "same":
		# padding_size = words_len - (words_len - n_gram_value + 1) = n_gram_value - 1
		padding_size = n_gram_value - 1
		# even size on left
		padding_left_size = np.ceil(padding_size / 2).astype(np.int32)
		# odd size on right
		padding_right_size = padding_size // 2
		words = np.pad(words, [(padding_left_size, padding_right_size)], "constant", constant_values=[0] * 2)
	else:
		raise Exception("Unkown padding mode")
	n_grams = list(zip(*[words[i:] for i in range(n_gram_value)]))
	if strides > 1:
		_n_grams = n_grams
		n_grams = []
		for gram_i in range(0, len(_n_grams), strides):
			n_grams.append(_n_grams[gram_i])
	return n_grams

class SamplePool(metaclass=ABCMeta):
	__metaclass__ = ABCMeta
	
	def __init__(self, chunk_size = 100):
		self.__chunk_size = chunk_size
	
	@property
	def chunk_size(self):
		return self.__chunk_size
	
	@abstractmethod
	def reset(self):
		pass
	
	def extend(self, samplepool_to_extend):
		if not isinstance(samplepool_to_extend, SamplePool):
			raise Exception("Illegal pool type")
	
	@abstractmethod
	def __next__(self):
		pass

class SimpleInMemorySamplePool(SamplePool):
	def __init__(self, samples, chunk_size):
		super(SimpleInMemorySamplePool, self).__init__(chunk_size)
		self.samples = samples
		self.iter_index = MutexVariable(0)

	def reset(self):
		self.iter_index = MutexVariable(0)

	def extend(self, samplepool_to_extend):
		super(SimpleInMemorySamplePool, self).extend(samplepool_to_extend)
		self.samples = np.concatenate([self.samples, samplepool_to_extend.samples])

	def __next__(self):
		chunk_size = self.chunk_size
		n_samples = len(self.samples)
		samples = self.samples

		self.iter_index.acquire()
		iter_index_value = self.iter_index.value
		new_iter_index_value = (iter_index_value + chunk_size) % n_samples
		self.iter_index.value = new_iter_index_value
		self.iter_index.release()

		if iter_index_value + chunk_size <= n_samples:
			batch_samples = samples[iter_index_value: iter_index_value + chunk_size]
			chunk_indices_list = np.array(list(range(iter_index_value, iter_index_value + chunk_size)))
		else:
			iter_samples_1 = samples[iter_index_value:]
			chunk_indices_1 = list(range(iter_index_value, n_samples))
			iter_samples_2 = samples[:new_iter_index_value]
			chunk_indices_2 = list(range(0, new_iter_index_value))
			batch_samples = np.concatenate([iter_samples_1, iter_samples_2])
			chunk_indices_list = np.concatenate([chunk_indices_1, chunk_indices_2])
		return batch_samples, chunk_indices_list

class InMemorySamplePool(SamplePool):
	def __init__(self, samples, chunk_size = 1000):
		super(InMemorySamplePool, self).__init__(chunk_size=chunk_size)
		if not isinstance(samples, np.ndarray):
			raise Exception("samples must be an ndarray")
		self.samples = samples
		self.iter_index = 0
	
	def reset(self):
		self.iter_index = 0
	
	def extend(self, samplepool_to_extend):
		super(InMemorySamplePool, self).extend(samplepool_to_extend)
		self.samples = np.concatenate([self.samples, samplepool_to_extend.samples])
	
	def __next__(self, cut = False, samples = None, iter_index = None, return_chunk_indices_list = False):
		if samples is None:
			samples = self.samples
		if iter_index is None:
			iter_index = self.iter_index
		assert len(samples) >= self.chunk_size
		iter_index_mutex = None
		if isinstance(iter_index, MutexVariable):
			iter_index_mutex = iter_index
			iter_index = iter_index_mutex.value
		else:
			iter_index = iter_index
		if iter_index + self.chunk_size <= len(samples):
			new_iter_index = (iter_index + self.chunk_size) % len(samples)
			if iter_index_mutex is not None and isinstance(iter_index_mutex, MutexVariable):
				iter_index_mutex.value = new_iter_index
				iter_index_mutex.release()
			else:
				self.iter_index = new_iter_index
			iter_samples = samples[iter_index: iter_index + self.chunk_size]
			chunk_indices_list = np.array(list(range(iter_index, iter_index + self.chunk_size)))
		else:
			if not cut:
				new_iter_index = (iter_index + self.chunk_size) % len(samples)
				if iter_index_mutex is not None and isinstance(iter_index_mutex, MutexVariable):
					iter_index_mutex.value = new_iter_index
					iter_index_mutex.release()
				else:
					self.iter_index = new_iter_index
				iter_samples_1 = samples[iter_index:]
				chunk_indices_1 = list(range(iter_index, len(samples)))
				iter_index = new_iter_index
				iter_samples_2 = samples[:iter_index]
				chunk_indices_2 = list(range(0, iter_index))
				iter_samples = np.concatenate([iter_samples_1, iter_samples_2]) 
				chunk_indices = np.concatenate([chunk_indices_1, chunk_indices_2]) 
				chunk_indices_list = chunk_indices
			else:
				if iter_index_mutex is not None and isinstance(iter_index_mutex, MutexVariable):
					iter_index_mutex.value = 0
					iter_index_mutex.release()
				else:
					self.iter_index = 0
				chunk_indices_list = np.zeros(len(samples) - iter_index, dtype=np.int32)
				iter_samples = samples[iter_index:]
				chunk_indices_list[:] = list(range(iter_index, len(samples)))
		if not return_chunk_indices_list:
			self.chunk_indices_list = chunk_indices_list
			return iter_samples
		else:
			return iter_samples, chunk_indices_list

# cut long sequence into multi subsequenced with length unroll_num
class Unroller(object):
	def __init__(self, X, y = None, unroll_num = 400, padding_val = 0):
		self.X = []
		self.y = []
		self.unroll_num = unroll_num
		self.X_sequence_len = []
		for x_i, x in enumerate(X):
			x_len = len(x)
			normalized_x_len = self.unroll_num * np.ceil(x_len / self.unroll_num).astype(np.int32)
			for start_i in range(0, normalized_x_len, unroll_num):
				x_slice = x[start_i: start_i + unroll_num]
				self.X_sequence_len.append(len(x_slice))
				if len(x_slice) < unroll_num:
					padding_size = unroll_num - len(x_slice)
					x_slice = np.pad(x_slice, [(0, padding_size)], "constant", constant_values=[padding_val] * 2)
				else:
					x_slice = np.array(x_slice)
				self.X.append(x_slice)
				if y is not None:
					self.y.append(y[x_i])
		if y is None:
			self.y = None


class AutoPaddingInMemorySamplePool(InMemorySamplePool):
	def __init__(self, samples, bins_count, batch_size=1000, pading_val=0, unkown_word_indicator=0, mode="random", y_is_sequence = False,
				 cut=True, roulette_cycle=None, y=None, get_y_in_batch=True, get_sequence_len_in_batch = True, unroll_num = -1):
		super(AutoPaddingInMemorySamplePool, self).__init__(samples=np.ndarray(1), chunk_size=batch_size)
		self.bins_count = bins_count
		self.pading_val = pading_val
		self.unkown_word_indicator = unkown_word_indicator
		self.iter_index_map = dict([(i, MutexVariable(0, name="iter_index_%s" % i)) for i in range(self.bins_count)])
		assert mode in ["random", "specific"]
		self.mode = mode
		self.cut = cut
		self.roulette_cycle = roulette_cycle
		self.y = y
		self._sorted_y = None
		self.get_y_in_batch = get_y_in_batch
		self.get_sequence_len_in_batch = get_sequence_len_in_batch
		self._sequence_len = None
		self.unroll_num = unroll_num
		self.y_is_sequence = y_is_sequence
		if unroll_num > 0:
			if not y_is_sequence:
				unroller = Unroller(samples, y, unroll_num=unroll_num)
				samples = unroller.X
				self.y = unroller.y
				self._sequence_len = unroller.X_sequence_len
			else:
				unroller = Unroller(samples, None, unroll_num=unroll_num)
				samples = unroller.X
				unroller = Unroller(y, None, unroll_num=unroll_num)
				self.y = unroller.X
				self._sequence_len = unroller.X_sequence_len

		self.build(samples)

	def build(self, samples):
		if not isinstance(samples, np.ndarray):
			samples = np.array(samples)
		self._samples = samples
		try:
			lens = np.array([len(x) for x in samples])
		except:
			lens = np.array([x.shape[0] for x in samples])
		self.sorted_indices = np.argsort(-lens)
		lens = lens[self.sorted_indices]
		n_samples = self._samples.shape[0]
		self.bins_bucket_edges = list(range(0, n_samples, int(n_samples / self.bins_count)))
		if len(self.bins_bucket_edges) == self.bins_count + 1:
			self.bins_bucket_edges[-1] = n_samples
		else:
			self.bins_bucket_edges.append(n_samples)
		self.min_gap = min([self.bins_bucket_edges[i + 1] - self.bins_bucket_edges[i] for i in range(self.bins_count)])
		self.bins_lens = [np.max(lens[self.bins_bucket_edges[i]: self.bins_bucket_edges[i + 1]]) for i in range(self.bins_count)]
		self.choice_roulette = reduce(lambda x, y: x + y, [[i] * np.ceil((self.bins_bucket_edges[i + 1] - self.bins_bucket_edges[i]) / self.chunk_size).astype(np.int32) for i in range(self.bins_count)])
		self.choice_index = MutexVariable(0, name="choice_index")
		if self.roulette_cycle is None:
			self.steps = len(self.choice_roulette)
		else:
			self.steps = len(self.roulette_cycle)

	def reset(self):
		self.iter_index_map = dict([(i, MutexVariable(0, name="iter_index_%s" % i)) for i in range(self.bins_count)])
		self.choice_index = MutexVariable(0, name="choice_index")

	def extend(self, samplepool_to_extend):
		super(AutoPaddingInMemorySamplePool, self).extend(samplepool_to_extend)
		if isinstance(samplepool_to_extend, AutoPaddingInMemorySamplePool):
			self.build(np.concatenate([self._samples, samplepool_to_extend._samples]))
		else:
			self.build(np.concatenate([self._samples, samplepool_to_extend.samples]))

	@property
	def sorted_samples(self):
		return self._samples[self.sorted_indices]

	@property
	def sorted_y(self):
		if self.y is None:
			raise Exception("y is None")
		if self._sorted_y is None:
			self._sorted_y = np.array(self.y, dtype=np.int32)[self.sorted_indices]
		return self._sorted_y

	@property
	def sequence_len(self):
		if self._sequence_len is None:
			raise Exception("sequence_len is None, only available when unroll_num > 0")
		if isinstance(self._sequence_len, list):
			self._sequence_len = np.array(self._sequence_len)[self.sorted_indices]
		return self._sequence_len

	# parallel on pool level, serial on bucket level
	def __next__(self):
		if self.mode == "random":
			start_index_i = np.random.choice(self.bins_count, 1)[0]
			cut = False
		else:
			if self.roulette_cycle is None:
				self.choice_index.acquire()
				start_index_i = self.choice_roulette[self.choice_index.value]
				self.choice_index.value = (self.choice_index + 1) % len(self.choice_roulette)
				self.choice_index.release()
			else:
				self.choice_index.acquire()
				start_index_i = self.choice_roulette[self.roulette_cycle[self.choice_index.value]]
				self.choice_index.value = (self.choice_index + 1) % len(self.roulette_cycle)
				self.choice_index.release()
			cut = self.cut
		end_index_i = start_index_i + 1
		start_index = self.bins_bucket_edges[start_index_i]
		end_index = self.bins_bucket_edges[end_index_i]
		samples = self._samples[self.sorted_indices[start_index: end_index]]
		if isinstance(self.iter_index_map[start_index_i], MutexVariable):
			# release in __next__
			self.iter_index_map[start_index_i].acquire()
		iter_index = self.iter_index_map[start_index_i]
		batch_samples, chunk_indices_list = super(AutoPaddingInMemorySamplePool, self).__next__(cut=cut,
																								samples=samples,
																								iter_index=iter_index,
																								return_chunk_indices_list=True)
		self.iter_index_map[start_index_i] = iter_index
		batch_len = self.bins_lens[start_index_i]
		if isinstance(batch_samples[0], (coo_matrix, csr_matrix, csc_matrix)):
			return_batch_samples = np.zeros((len(batch_samples), batch_len, batch_samples[0].shape[1]), dtype=np.int32)
		else:
			return_batch_samples = np.zeros((len(batch_samples), batch_len), dtype=np.int32)
		cur_batch_sequence_len = np.zeros(len(batch_samples))
		for i, sample in enumerate(batch_samples):
			if not isinstance(sample, (coo_matrix, csr_matrix, csc_matrix)):
				if len(sample) < batch_len:
					padding_size = batch_len - len(sample)
					sample = np.pad(sample, [(0, padding_size)], "constant", constant_values=[self.unkown_word_indicator] * 2)
				return_batch_samples[i] = sample
				if self.unroll_num < 0:
					cur_batch_sequence_len[i] = len(sample)
			elif isinstance(sample, (coo_matrix, csr_matrix, csc_matrix)):
				return_batch_samples[i, sample.row, sample.col] = sample.data # 1
				if self.unroll_num < 0:
					cur_batch_sequence_len[i] = len(sample.data)
		chunk_indices_list += start_index
		# unroll > 0
		if self.unroll_num > 0:
			cur_batch_sequence_len = self.sequence_len[chunk_indices_list]
		return_tuple = [return_batch_samples]
		if self.get_y_in_batch and self.y:
			return_tuple.append(self.sorted_y[chunk_indices_list])
		if self.get_sequence_len_in_batch:
			return_tuple.append(cur_batch_sequence_len.reshape((-1, 1)))
		return return_tuple
		

# file must be saved as csv with header 
class OutMemorySamplePool(SamplePool):
	def __init__(self, sample_filepath_list, target_col_name, chunk_size = 1000):
		super(OutMemorySamplePool, self).__init__(chunk_size=chunk_size)
		if sample_filepath_list is None or not isinstance(sample_filepath_list, list) or len(sample_filepath_list) == 0:
			raise Exception("filepath list is empty or not a list")
		self.sample_filepath_list = sample_filepath_list
		self.target_col_name = target_col_name
		self.dataframe_iter = None
		self.dataframe_iter_index = 0
		self.__current_chunk = None
		self.chunk_cumulation = 0
	
	def reset(self):
		self.dataframe_iter = None
		self.dataframe_iter_index = 0
		self.__current_chunk = None
	
	def extend(self, samplepool_to_extend):
		super(OutMemorySamplePool, self).extend(samplepool_to_extend)
		self.sample_filepath_list.extend(samplepool_to_extend.sample_filepath_list)
	
	def __get_chunk__(self):
		current_chunk_size = None
		extra_chunk_size = -1
		while 1:
			if self.dataframe_iter is None:
				self.dataframe_iter = pd.read_csv(self.sample_filepath_list[self.dataframe_iter_index], usecols=[self.target_col_name], iterator=True, chunksize=self.chunk_size, encoding="utf-8")
				self.dataframe_iter_index = (self.dataframe_iter_index + 1) % len(self.sample_filepath_list)
			try:
				if self.__current_chunk is not None:
					extra_chunk_size = self.chunk_size - len(self.__current_chunk)
					extra_chunk = self.dataframe_iter.get_chunk(extra_chunk_size).to_dict("list")[self.target_col_name]
					current_chunk = np.concatenate([self.__current_chunk, extra_chunk])
					current_chunk_size = len(self.__current_chunk)
					self.__current_chunk = None
				else:
					current_chunk = self.dataframe_iter.get_chunk().to_dict("list")[self.target_col_name]
				if len(current_chunk) < self.chunk_size:
					self.__current_chunk = current_chunk
					self.dataframe_iter = None
					continue
				else:
					if self.dataframe_iter_index - 1 == 0 and current_chunk_size is not None:
						chunk_indices_1 = list(range(self.chunk_cumulation, self.chunk_cumulation + current_chunk_size))
						chunk_indices_2 = list(range(0, extra_chunk_size))
						self.chunk_indices_list[:] = np.concatenate([chunk_indices_1, chunk_indices_2])
						self.chunk_cumulation = extra_chunk_size
						current_chunk_size = None
					else:
						self.chunk_indices_list[:] = list(range(self.chunk_cumulation, self.chunk_cumulation + self.chunk_size))
						self.chunk_cumulation += self.chunk_size
			except StopIteration:
				self.dataframe_iter.close()
				self.dataframe_iter = None
				if self.dataframe_iter_index == 0:
					self.chunk_cumulation = 0
				continue
			return current_chunk
	
	def __next__(self):
		return self.__get_chunk__()

class WordCounter(object):
	def __init__(self, seed = 0, lower_case = False):
		self.tok = nltk.tokenize.toktok.ToktokTokenizer()
		self.words_list = []
		self.target_col = None
		self.random =  np.random.RandomState(seed)
		self.lower_case = lower_case
		self.unk_tag = "unk"
		self.bos_tag = "<bos>"
		self.eos_tag = "</eos>"
		self.unk_index = 0
		self.bos_index = 1
		self.eos_index = 2
		self._default_counts = [["unk", -1], [self.bos_tag, -1], [self.eos_tag, -1]]
		self.default_counts_len = len(self.default_counts)

	def load(self, datapath):
		with open(datapath, "rb") as f:
			self.words_list = pickle.load(f)
		return self

	@property
	def default_counts(self):
		return copy.deepcopy(self._default_counts)
		
	def __check_filepath_list(self, filepath_list):
		if filepath_list is None or not isinstance(filepath_list, list) or len(filepath_list) == 0:
			raise Exception("filepath list is empty or not a list")

	def __clean_text(self, content):
		
		content = content.replace('\n',' ').replace('<br />', ' ')
		# punctuation = """"""
		# for p in punctuation:
		# 	content = content.replace(p, " %s " % p)
		content = re.sub(r"([.,?!:;(){}\[\]])", r" \1 ", content)
		return content
	
	def fit(self, filepath_list, target_col = None, clean_text_func = None, doc_count_threshold = None):
		self.__check_filepath_list(filepath_list)
		self.target_col = target_col
		counter = defaultdict(lambda: 0)
		if clean_text_func is None:
			clean_text_func = self.__clean_text
		doc_counts = defaultdict(int)
		for filepath in filepath_list:
			if self.target_col is not None:
				data_df_iter = pd.read_csv(filepath, iterator=True, usecols=[self.target_col], chunksize=100000, encoding="utf-8")
				for chunk in data_df_iter:
					for content in chunk[target_col]:
						doc_seen = set()
						content = clean_text_func(content)
						words = [w.lower() if self.lower_case else w for w in self.tok.tokenize(content)]
						for word in words:
							counter[word] += 1
							if word not in doc_seen:
								doc_counts[word] += 1
								doc_seen.add(word)
			else:
				data_iter = open(filepath, 'r', encoding="utf-8")
				for content in data_iter:
					doc_seen = set()
					content = clean_text_func(content)
					words = [w.lower() if self.lower_case else w for w in self.tok.tokenize(content)]
					for word in words:
						counter[word] += 1
						if word not in doc_seen:
							doc_counts[word] += 1
							doc_seen.add(word)
		if doc_count_threshold is not None:
			self.words_list = [(term, freq) for term, freq in counter.items() if doc_counts[term] > doc_count_threshold]
		else:
			self.words_list = list(counter.items())
		self.words_list.sort(key=lambda x: -x[1])

	# higher the frequency, easier to erase out
	def getWordsStatistics(self, sentences, sample = 1e-4):
		WordStat = namedtuple("WordStat", ["count", "sample_int"])
		wordsStat = defaultdict(lambda : WordStat(count=0, sample_int=sample))
		retain_total = 0
		for sent in sentences:
			for word in sent:
				wordsStat[word].count += 1
				retain_total += 1
		threshold_count = sample * retain_total
		for w in wordsStat:
			v = wordsStat[w].count
			word_probability = (np.sqrt(v / threshold_count) + 1) * (threshold_count / v) # p ~= sqrt(threshold_count / count), count up, p down
			if word_probability > 1.0:
				word_probability = 1.0
			wordsStat[w].sample_int = int(round(word_probability * 2**32))
		return wordsStat
	
	def most_common(self, vocab_size):
		return self.words_list[:vocab_size]

	def most_common_freqs(self, max_words):
		vocab_size = max_words - len(self._default_counts)
		assert vocab_size >= 0, "vocab_size must greater than or equal to 0"
		freqs = list(map(lambda x: x[1], self.words_list[:vocab_size]))
		freqs = [1] * len(self._default_counts) + freqs
		return freqs
	
	def transform(self, filepath_list, max_words = None, clean_text_func = None, start_ch = None, include_unk = False):
		self.__check_filepath_list(filepath_list)
		if clean_text_func is None:
			clean_text_func = self.__clean_text
		counts = self.default_counts
		if max_words is None:
			max_words = len(self.words_list) + self.default_counts_len
		counts.extend(self.most_common(max_words - self.default_counts_len))
		dictionary = {}
		documents_indices = []
		for word, _ in counts:
			dictionary[word] = len(dictionary)
		for filepath in filepath_list:
			if self.target_col is not None:
				data_df_iter = pd.read_csv(filepath, iterator=True, usecols=[self.target_col], chunksize=100000, encoding="utf-8")
				for chunk in data_df_iter:
					for content in chunk[self.target_col]:
						content = clean_text_func(content)
						words = [w.lower() if self.lower_case else w for w in self.tok.tokenize(content)]
						word_indices = [self.bos_index]
						if start_ch is not None:
							word_indices.append(start_ch)
						for word in words:
							if word in dictionary:
								index = dictionary[word]
								word_indices.append(index)
							else:
								if include_unk:
									index = 0
									word_indices.append(index)
						word_indices.append(self.eos_index)
						documents_indices.append(word_indices)
			else:
				data_iter = open(filepath, 'r', encoding="utf-8")
				for content in data_iter:
					content = clean_text_func(content)
					words = [w.lower() if self.lower_case else w for w in self.tok.tokenize(content)]
					word_indices = [self.bos_index]
					if start_ch is not None:
						word_indices.append(start_ch)
					for word in words:
						if word in dictionary:
							index = dictionary[word]
							word_indices.append(index)
						else:
							if include_unk:
								index = 0
								word_indices.append(index)
					word_indices.append(self.eos_index)
					documents_indices.append(word_indices)
		return documents_indices

	def transform_docs(self, docs, max_words = None, clean_text_func = None, start_ch = None, include_unk = False):
		if clean_text_func is None:
			clean_text_func = self.__clean_text
		counts = self.default_counts
		if max_words is None:
			max_words = len(self.words_list) + self.default_counts_len
		counts.extend(self.most_common(max_words - self.default_counts_len))
		dictionary = {}
		documents_indices = []
		for word, _ in counts:
			dictionary[word] = len(dictionary)
		for content in docs:
			content = clean_text_func(content)
			words = [w.lower() for w in self.tok.tokenize(content)]
			word_indices = [self.bos_index]
			if start_ch is not None:
				word_indices.append(start_ch)
			for word in words:
				if word in dictionary:
					index = dictionary[word]
					word_indices.append(index)
				else:
					if include_unk:
						index = 0
						word_indices.append(index)
				word_indices.append(self.eos_index)
			documents_indices.append(word_indices)
				
		return documents_indices

	# indices to words
	def reverse(self, indices,  num_words = None, ignore_freq_than = 1000000000, wordsStat = None, return_list = False):
		counts = self.default_counts
		if num_words is None:
			num_words = len(self.words_list) + self.default_counts_len
		counts.extend(self.most_common(num_words - self.default_counts_len))
		dictionary = {}
		freq_dist = {}
		for word, freq in counts:
			freq_dist[len(dictionary)] = freq
			dictionary[len(dictionary)] = word
		docs = []
		for i, doc_indices in enumerate(indices):
			doc_words = []
			# erase high frequency words
			if wordsStat is not None:
				doc_indices = [w for w in doc_indices if w in wordsStat and wordsStat[w].sample_int > self.random.rand() * 2 ** 32]
			for word_index in doc_indices:
				if word_index in dictionary and freq_dist[word_index] <= ignore_freq_than:
					doc_words.append(dictionary[word_index])
				else:
					doc_words.append(dictionary[0])
			if not return_list:
				docs.append(" ".join(doc_words))
			else:
				docs.append(doc_words)
		return docs

	def get_i2w_dictionary(self, num_words = None):
		counts = self.default_counts
		if num_words is None:
			num_words = len(self.words_list) + self.default_counts_len
		counts.extend(self.most_common(num_words - self.default_counts_len))
		dictionary = {}
		for word, freq in counts:
			dictionary[len(dictionary)] = word
		return dictionary

	def get_w2i_dictionary(self, num_words = None):
		counts = self.default_counts
		if num_words is None:
			num_words = len(self.words_list) + self.default_counts_len
		counts.extend(self.most_common(num_words - self.default_counts_len))
		dictionary = {}
		for word, freq in counts:
			dictionary[word] = len(dictionary)
		return dictionary

	@classmethod
	def one_hot(cls, n_vocab, documents_indices):
		n_docs, n_words = documents_indices.shape
		doc_onehot = []
		for i, doc in enumerate(documents_indices):
			rows_cols_data = [(j, doc[j], 1) for j in range(len(doc))]
			coomat = coo_matrix((rows_cols_data[:, 2], (rows_cols_data[:, 0], rows_cols_data[:, 1])), shape=(n_words, n_vocab))
			doc_onehot.append(coomat)
		return doc_onehot

	@classmethod
	def ngram_one_hot(cls, n_vocab, documents_indices, n_gram_value = 2, cumulate_add = False):
		n_docs, n_words = documents_indices.shape
		doc_onehot = []
		for i, doc in enumerate(documents_indices):
			doc_n_grams = zip(*[doc[i:] for i in range(n_gram_value)])
			doc_n_grams = reduce(lambda x, y: x + y, doc_n_grams)
			rows_cols_data = [(j // n_gram_value, doc_n_grams[j], 1) for j in range(len(doc_n_grams))]
			if not cumulate_add:
				rows_cols_data = set(rows_cols_data)
			rows_cols_data = np.array(list(rows_cols_data))
			coomat = coo_matrix((rows_cols_data[:, 2], (rows_cols_data[:, 0], rows_cols_data[:, 1])), shape=(n_words - n_gram_value + 1, n_vocab))
			doc_onehot.append(coomat)
		return doc_onehot

	def get_pretrain_embedding(self, model, num_words = None, size = 300):
		words_matrix = self.random.rand(num_words, size)
		counts = self.default_counts
		if num_words is None:
			num_words = len(self.words_list) + self.default_counts_len
		counts.extend(self.most_common(num_words - self.default_counts_len))
		dictionary = {}
		for word, _ in counts:
			dictionary[word] = len(dictionary)
		num_not_in = 0
		not_in_list = []
		for word, index in dictionary.items():
			if word in model.wv:
				words_matrix[index][:] = model.wv[word]
			else:
				num_not_in += 1
				not_in_list.append(word)
		return words_matrix, not_in_list