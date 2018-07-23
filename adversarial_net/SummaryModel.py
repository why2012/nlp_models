from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
from adversarial_net.inputs import DataLoader, getDatasetFilePath
from adversarial_net.inputs import construct_summary_model_bucket_input
from adversarial_net.engine import BaseModel
from adversarial_net import arguments as flags
from adversarial_net import sequences as seq
from adversarial_net import layers
from adversarial_net.utils import getLogger
from adversarial_net import osp

logger = getLogger("summary_model")
SOS_TAG = 1
EOS_TAG = 2

class SummaryModel(BaseModel):
    def __init__(self, use_average = False):
        super(SummaryModel, self).__init__(use_average=use_average)
        self.to_embedding = seq.EmbeddingSequence(
            var_scope_name="embedding",
            vocab_size=self.arguments["lm_sequence"]["vocab_size"],
            embedding_dim=self.arguments["lm_sequence"]["embedding_dim"],
            vocab_freqs=self.arguments["vocab_freqs"],
            normalize=self.arguments["lm_sequence"]["normalize"],
            keep_embed_prob=self.arguments["lm_sequence"]["keep_embed_prob"])
        # self.to_embedding_decoder = seq.EmbeddingSequence(
        #     var_scope_name="embedding_decoder",
        #     vocab_size=self.arguments["lm_sequence"]["vocab_size"],
        #     embedding_dim=self.arguments["lm_sequence"]["embedding_dim"],
        #     vocab_freqs=self.arguments["vocab_freqs"],
        #     normalize=self.arguments["lm_sequence"]["normalize"],
        #     keep_embed_prob=self.arguments["lm_sequence"]["keep_embed_prob"])
        self.to_embedding_decoder = self.to_embedding
        self.grus = seq.SummaryGRUs(var_scope_name="GRUs",
                                    state_size=self.arguments["summary"]["rnn_cell_size"],
                                    input_dim=self.arguments["lm_sequence"]["embedding_dim"],
                                    gru_keep_prob_out=self.arguments["summary"]["rnn_keep_prob_out"],
                                    build=False)
        self.atten_loss = seq.SummaryBahdanauAttentionLoss(
            var_scope_name="BahdanauAttentionLoss",
            encoder_fw_cell=self.grus.encoder_fw_cell,
            encoder_bw_cell=self.grus.encoder_bw_cell,
            decoder_cell=self.grus.decoder_cell,
            rnn_size=self.arguments["summary"]["rnn_cell_size"],
            vocab_size=self.arguments["lm_sequence"]["vocab_size"],
            num_candidate_samples=self.arguments["lm_loss"]["num_candidate_samples"],
            vocab_freqs=self.arguments["vocab_freqs"])

        self.eval_layer = seq.EvalSummaryBahdanauAttention(
            associate_var_scope_name="BahdanauAttentionLoss",
            encoder_fw_cell=self.grus.encoder_fw_cell,
            encoder_bw_cell=self.grus.encoder_bw_cell,
            decoder_cell=self.grus.decoder_cell,
            state_proj_layer=self.atten_loss.state_proj_layer,
            to_embedding_layers=self.to_embedding,
            to_embedding_layers_decoder=self.to_embedding_decoder,
            rnn_size=self.arguments["summary"]["rnn_cell_size"],
            vocab_size=self.arguments["lm_sequence"]["vocab_size"])

    def build(self):
        logger.info("constructing summary model dataset...")
        encoder_bucket, decoder_bucket = construct_summary_model_bucket_input(
            datapath=self.arguments["inputs"]["datapath"],
            dataset="summary",
            modelname="summary_model",
            batch_size=self.arguments["inputs"]["batch_size"],
            encoder_decoder_bucket_boundaries=[(30, 10), (50, 20), (70, 20), (100, 20), (200, 30)])
        logger.info("summary model dataset is constructed.")
        encoder_len_tensor = encoder_bucket["encoder_len"]
        encoder_input_tensor = encoder_bucket["encoder_input"]
        decoder_len_tensor = decoder_bucket["decoder_len"]
        decoder_input_tensor = decoder_bucket["decoder_input"]
        decoder_target_tensor = decoder_bucket["decoder_target"]
        encoder_embed_inputs = self.to_embedding(encoder_input_tensor)
        decoder_embed_inputs = self.to_embedding_decoder(decoder_input_tensor)
        self.loss, _ = self.atten_loss(
                                    encoder_embed_inputs=encoder_embed_inputs,
                                    decoder_embed_inputs=decoder_embed_inputs,
                                    decoder_targets=decoder_target_tensor,
                                    encoder_len=encoder_len_tensor,
                                    decoder_len=decoder_len_tensor)
        self.train_op = self.optimize(self.loss, self.arguments["max_grad_norm"],
                                      self.arguments["lr"], self.arguments["lr_decay"],
                                      norm_embedding = True, optimizer=tf.train.AdadeltaOptimizer,
                                      optimizer_kwargs={"epsilon": 1e-6})
        super(SummaryModel, self).build()

    def eval(self, inputs_docs, save_model_path, lower_case=True):
        batch_size = len(inputs_docs)
        wordCounter = DataLoader.reload_word_counter(
            vocab_abspath=getDatasetFilePath(self.arguments["inputs"]["datapath"], "summary", "word_freqs"))
        wordCounter.lower_case = lower_case
        inputs_docs_idx = wordCounter.transform_docs(docs=inputs_docs, max_words=self.arguments["lm_sequence"]["vocab_size"])
        max_len = max(map(len, inputs_docs_idx))
        seq_len = []
        for i, idx in enumerate(inputs_docs_idx):
            padding_size = max_len - len(idx)
            inputs_docs_idx[i] = np.pad(idx, [(0, padding_size)], "constant", constant_values=[0] * 2)
            seq_len.append(len(idx))
        to_embedding_layers_placeholder = tf.placeholder(tf.int32, shape=[None, max_len])
        seq_len_placeholder = tf.placeholder(tf.int32, shape=[None])
        outputs, final_sequence_lengths = self.eval_layer(batch_size=batch_size, sos_tag=SOS_TAG, eos_tag=EOS_TAG,
                                                          encoder_len=seq_len_placeholder,
                                                          encoder_embed_inputs=self.to_embedding(to_embedding_layers_placeholder),
                                                          beam_width=self.arguments["summary"]["beam_width"],
                                                          maximum_iterations=self.arguments["summary"]["maximum_iterations"])
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            self._resotre_training_model(sess=sess, save_model_path=save_model_path)
            output_idx, final_sequence_lengths_val = sess.run([outputs, final_sequence_lengths], feed_dict={
                to_embedding_layers_placeholder: inputs_docs_idx, seq_len_placeholder: seq_len})
        output_idx = output_idx[:, :, 0]
        final_sequence_lengths_val = final_sequence_lengths_val[:, 0]
        output_words = wordCounter.reverse(output_idx, self.arguments["lm_sequence"]["vocab_size"], return_list=True)
        inputs_docs = wordCounter.reverse(inputs_docs_idx, self.arguments["lm_sequence"]["vocab_size"])
        for i in range(len(inputs_docs)):
            logger.info("-"*20 + " doc-%s " % i + "-"*20)
            logger.info("doc: " + inputs_docs[i])
            logger.info("title: " + " ".join(output_words[i][:final_sequence_lengths_val[i]]))

    def fit(self, model_inpus = None, save_model_path = None, pretrained_model_path = None, remove_variable_scope_prefix = True):
        variables_to_restore = []
        for variable in tf.trainable_variables():
            if "embedding" in variable.op.name:
                if remove_variable_scope_prefix:
                    if isinstance(variables_to_restore, list):
                        variables_to_restore = {}
                    variables_to_restore[variable.op.name.split("/", 1)[1]] = variable
                else:
                    variables_to_restore.append(variable)

        super(SummaryModel, self)._fit(model_inpus, save_model_path, pretrained_model_path, variables_to_restore=variables_to_restore)



