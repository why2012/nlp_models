from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import pandas as pd
import numpy as np
import nltk
import pickle
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from functools import reduce

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
	
	def __next__(self, cut = False):
		assert len(self.samples) >= self.chunk_size
		if self.iter_index + self.chunk_size <= len(self.samples):
			iter_samples = self.samples[self.iter_index: self.iter_index + self.chunk_size]
			self.chunk_indices_list = np.array(list(range(self.iter_index, self.iter_index + self.chunk_size)))
			self.iter_index = (self.iter_index + self.chunk_size) % len(self.samples)
		else:
			if not cut:
				iter_samples_1 = self.samples[self.iter_index:]
				chunk_indices_1 = list(range(self.iter_index, len(self.samples)))
				self.iter_index = (self.iter_index + self.chunk_size) % len(self.samples)
				iter_samples_2 = self.samples[:self.iter_index]
				chunk_indices_2 = list(range(0, self.iter_index))
				iter_samples = np.concatenate([iter_samples_1, iter_samples_2]) 
				chunk_indices = np.concatenate([chunk_indices_1, chunk_indices_2]) 
				self.chunk_indices_list = chunk_indices
			else:
				self.chunk_indices_list = np.zeros(len(self.samples) - self.iter_index, dtype=np.int32)
				iter_samples = self.samples[self.iter_index:]
				self.chunk_indices_list[:] = list(range(self.iter_index, len(self.samples)))
				self.iter_index = 0
		return iter_samples

class AutoPaddingInMemorySamplePool(InMemorySamplePool):
	def __init__(self, samples, bins_count, chunk_size = 1000, pading_val = 0, unkown_word_indicator = 0, mode = "random", roulette_cycle = None):
		super(AutoPaddingInMemorySamplePool, self).__init__(samples=np.ndarray(1), chunk_size=chunk_size)
		self.bins_count = bins_count
		self.pading_val = pading_val
		self.unkown_word_indicator = unkown_word_indicator
		self.iter_index_map = defaultdict(int)
		assert mode in ["random", "specific"]
		self.mode = mode
		self.roulette_cycle = roulette_cycle
		self.build(samples)

	def build(self, samples):
		if not isinstance(samples, np.ndarray):
			samples = np.array(samples)
		self._samples = samples
		lens = np.array([len(x) for x in samples])
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
		self.choice_index = 0
		if self.roulette_cycle is None:
			self.steps = len(self.choice_roulette)
		else:
			self.steps = len(self.roulette_cycle)

	def reset(self):
		self.iter_index_map = defaultdict(int)
		self.choice_index = 0

	def extend(self, samplepool_to_extend):
		super(AutoPaddingInMemorySamplePool, self).extend(samplepool_to_extend)
		if isinstance(samplepool_to_extend, AutoPaddingInMemorySamplePool):
			self.build(np.concatenate([self._samples, samplepool_to_extend._samples]))
		else:
			self.build(np.concatenate([self._samples, samplepool_to_extend.samples]))

	def __next__(self):
		if self.mode == "random":
			start_index_i = np.random.choice(self.bins_count, 1)[0]
			cut = False
		else:
			if self.roulette_cycle is None:
				start_index_i = self.choice_roulette[self.choice_index]
				self.choice_index = (self.choice_index + 1) % len(self.choice_roulette)
			else:
				start_index_i = self.choice_roulette[self.roulette_cycle[self.choice_index]]
				self.choice_index = (self.choice_index + 1) % len(self.roulette_cycle)
			cut = True
		end_index_i = start_index_i + 1
		start_index = self.bins_bucket_edges[start_index_i]
		end_index = self.bins_bucket_edges[end_index_i]
		self.samples = self._samples[self.sorted_indices[start_index: end_index]]
		self.iter_index = self.iter_index_map[start_index_i]
		batch_samples = super(AutoPaddingInMemorySamplePool, self).__next__(cut=cut)
		self.iter_index_map[start_index_i] = self.iter_index
		batch_len = self.bins_lens[start_index_i]
		return_batch_samples = np.zeros((len(batch_samples), batch_len), dtype=np.int32)
		for i, sample in enumerate(batch_samples):
			if len(sample) < batch_len:
				padding_size = batch_len - len(sample)
				sample = np.pad(sample, [(0, padding_size)], "constant", constant_values=[self.unkown_word_indicator] * 2)
			return_batch_samples[i] = sample
		self.chunk_indices_list += start_index
		return return_batch_samples
		

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
		while(1):
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
	def __init__(self):
		self.tok = nltk.tokenize.toktok.ToktokTokenizer()
		self.words_list = []
		self.target_col = None
		
	def __check_filepath_list(self, filepath_list):
		if filepath_list is None or not isinstance(filepath_list, list) or len(filepath_list) == 0:
			raise Exception("filepath list is empty or not a list")

	def __clean_text(self, content):
		content = content.replace('\n',' ').replace('<br />', ' ')
		return content
	
	def fit(self, filepath_list, target_col = None, clean_text_func = None):
		self.__check_filepath_list(filepath_list)
		self.target_col = target_col
		counter = defaultdict(lambda: 0)
		if clean_text_func is None:
			clean_text_func = self.__clean_text
		for filepath in filepath_list:
			if self.target_col is not None:
				data_df_iter = pd.read_csv(filepath, iterator=True, usecols=[self.target_col], chunksize=100000, encoding="utf-8")
				for chunk in data_df_iter:
					for content in chunk[target_col]:
						content = clean_text_func(content)
						words = [w.lower() for w in self.tok.tokenize(content)]
						for word in words:
							counter[word] += 1
			else:
				data_iter = open(filepath, 'r', encoding="utf-8")
				for content in data_iter:
					content = clean_text_func(content)
					words = [w.lower() for w in self.tok.tokenize(content)]
					for word in words:
						counter[word] += 1
		self.words_list = list(counter.items())
		self.words_list.sort(key=lambda x: -x[1])
	
	def most_common(self, vocab_size):
		return self.words_list[:vocab_size]
	
	def transform(self, filepath_list, max_words = None, clean_text_func = None):
		self.__check_filepath_list(filepath_list)
		if clean_text_func is None:
			clean_text_func = self.__clean_text
		counts = [["unk", -1]]
		if max_words is None:
			max_words = len(self.words_list) + 1
		counts.extend(self.most_common(max_words - 1))
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
						words = [w.lower() for w in self.tok.tokenize(content)]
						word_indices = []
						for word in words:
							if word in dictionary:
								index = dictionary[word]
							else:
								index = 0
							word_indices.append(index)
						documents_indices.append(word_indices)
			else:
				data_iter = open(filepath, 'r', encoding="utf-8")
				for content in data_iter:
					content = clean_text_func(content)
					words = [w.lower() for w in self.tok.tokenize(content)]
					word_indices = []
					for word in words:
						if word in dictionary:
							index = dictionary[word]
						else:
							index = 0
						word_indices.append(index)
					documents_indices.append(word_indices)
		return documents_indices

	def get_pretrain_embedding(self, model, num_words = None, size = 200):
	    words_matrix = np.random.randn(num_words, size)
	    counts = [["unk", -1]]
	    if num_words is None:
	        num_words = len(self.words_list) + 1
	    counts.extend(self.most_common(num_words - 1))
	    dictionary = {}
	    documents_indices = []
	    for word, _ in counts:
	        dictionary[word] = len(dictionary)
	    for word, index in dictionary.items():
	        if word in model.wv:
	            words_matrix[index][:] = model.wv[word]
	    return words_matrix