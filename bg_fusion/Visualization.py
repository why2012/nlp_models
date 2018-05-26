from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import keras.backend as K
from keras.preprocessing import sequence
import numpy as np
import preprocessing

def get_layer_activations(model, layer_index, input_ndarray, need_padding = False, maxlen = None, input_layer_index = 0, output_layer_index = -1):
	if need_padding:
		if maxlen is None:
			maxlen = [len(x) for x in input_ndarray]
		input_ndarray = sequence.pad_sequences(input_ndarray, maxlen=maxlen, padding='post', truncating='post')
	intput_placeholder = model.layers[input_layer_index].input
	output_tensor = model.layers[output_layer_index].output
	layer_output_tensor = model.layers[layer_index].output
	layer_output, output_values = K.get_session().run([layer_output_tensor, output_tensor], feed_dict={intput_placeholder: input_ndarray})
	return layer_output, output_values

def view_n_gram_activations(layer_output, output_values, docs_indices, wordCounter, num_words, n_gram_value = 1, n_gram_stride = 1, padding = "valid", top_n = 20, tail_n = 0, print_words = False, y_set = None):
	n_docs = len(docs_indices)
	layer_words = wordCounter.reverse(docs_indices, num_words=num_words, return_list=True)
	for doc_i in range(n_docs):
		doc_layer_words = layer_words[doc_i]
		if y_set is not None:
			y_real = y_set[doc_i]
		if n_gram_value > 1:
			doc_layer_words = preprocessing.make_n_gram(doc_layer_words, n_gram_value=n_gram_value, strides=n_gram_stride, padding=padding)
		doc_layer_words = np.array(doc_layer_words)
		doc_layer_output = layer_output[doc_i]
		doc_output_values = output_values[doc_i]
		layer_output_sum = np.sum(doc_layer_output, axis=1)
		top_n_index = np.argsort(-layer_output_sum)[:top_n].tolist()
		if tail_n > 0:
			top_n_index_r = np.argsort(layer_output_sum)[:tail_n].tolist()
			top_n_index = top_n_index + top_n_index_r
		print("===doc-%s===" % doc_i, "[Words Activations]: ", layer_output_sum[top_n_index].tolist())
		print("===doc-%s===" % doc_i, "[Top %s Grams]: " % (top_n + tail_n), doc_layer_words[top_n_index[:top_n]].tolist(), doc_layer_words[top_n_index[top_n:]].tolist())
		print("===doc-%s===" % doc_i, "[Predict Label]: ", doc_output_values)
		if y_set is not None:
			print("===doc-%s===" % doc_i, "[Real Label]: ", y_real)
		if print_words:
			# if not isinstance(layer_words[0], str):
			# 	print("===doc-%s===" % doc_i, "[Words]: ", " ".join([str(w) for w in layer_words]))
			# else:
			# 	print("===doc-%s===" % doc_i, "[Words]: ", " ".join(layer_words))
			print("===doc-%s===" % doc_i, "[Words]: ", " ".join(layer_words[doc_i]))