from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
from adversarial_net.engine import BaseModel
from adversarial_net import arguments as flags
from adversarial_net import sequences as seq
from adversarial_net import layers
from adversarial_net.utils import getLogger
from adversarial_net.inputs import construct_language_model_input_tensor_with_state
from adversarial_net.inputs import construct_classification_model_input_tensor_with_state
from adversarial_net.inputs import construct_autoencoder_model_input_tensor_with_state
from adversarial_net import osp

logger = getLogger("model")

class LanguageModel(BaseModel):
    def __init__(self, use_average = False):
        super(LanguageModel, self).__init__(use_average=use_average)
        logger.info("constructing language model dataset...")
        self.inputs, self.get_lstm_state, self.save_lstm_state = construct_language_model_input_tensor_with_state(**self.arguments["lm_inputs"])
        logger.info("language model dataset is constructed.")
        self.sequences["lm_sequence"] = seq.LanguageModelSequence(**self.arguments["lm_sequence"])
        self.loss_layer = layers.SoftmaxLoss(**self.arguments["lm_loss"])
        self.train_op = None
        self.loss = None
        self.acc = None
        lm_sequence = self.arguments["lm_sequence"].copy()
        del lm_sequence["vocab_freqs"]
        lm_loss = self.arguments["lm_loss"].copy()
        del lm_loss["vocab_freqs"]
        print("lm_inputs", self.arguments["lm_inputs"])
        print("lm_sequence", lm_sequence)
        print("lm_loss", lm_loss)

    def build(self):
        X_tensor, y_tensor, weight_tensor = tf.squeeze(self.inputs.sequences["X"], axis=-1), tf.squeeze(
            self.inputs.sequences["y"], axis=-1), tf.squeeze(self.inputs.sequences["weight"], axis=-1)
        lstm_initial_state = self.get_lstm_state()
        output_tensor, final_state = self.sequences["lm_sequence"](X_tensor, lstm_initial_state, sequence_len=self.inputs.length)
        self.loss = self.loss_layer((output_tensor, y_tensor, weight_tensor))
        self.acc = self.loss_layer.lm_acc
        with tf.control_dependencies([self.save_lstm_state(final_state)]):
            self.loss = tf.identity(self.loss)
        self.train_op = self.optimize(self.loss, self.arguments["max_grad_norm"],
                                      self.arguments["lr"], self.arguments["lr_decay"])
        super(LanguageModel, self).build()

    def fit(self, model_inpus = None, save_model_path = None, pretrained_model_path = None):
        variables_to_restore = self.trainable_weights()
        super(LanguageModel, self)._fit(model_inpus, save_model_path, pretrained_model_path, variables_to_restore=variables_to_restore)

class AutoEncoderModel(BaseModel):
    def __init__(self, use_average = False):
        super(AutoEncoderModel, self).__init__(use_average=use_average)
        logger.info("constructing auto encoder model dataset...")
        self.inputs, self.get_lstm_state, self.save_lstm_state = construct_autoencoder_model_input_tensor_with_state(**self.arguments["ae_inputs"])
        logger.info("encoder model dataset is constructed.")
        # same structure with language model
        self.sequences["ae_sequence"] = seq.LanguageModelSequence(**self.arguments["ae_sequence"])
        self.loss_layer = layers.SoftmaxLoss(**self.arguments["ae_loss"])
        self.train_op = None
        self.loss = None
        self.acc = None

    def build(self):
        # X_tensor (None, steps)
        # y_tensor (None, steps)
        # weight_tensor (None, steps)
        X_tensor, y_tensor, weight_tensor = tf.squeeze(self.inputs.sequences["X"], axis=-1), tf.squeeze(
            self.inputs.sequences["y"], axis=-1), tf.squeeze(self.inputs.sequences["weight"], axis=-1)
        # [LSTMTuple(c, h), ]
        lstm_initial_state = self.get_lstm_state()
        # output_tensor (None, steps, lstm_size)
        # final_state (None, lstm_size)
        output_tensor, final_state = self.sequences["ae_sequence"](X_tensor, lstm_initial_state, sequence_len=self.inputs.length)
        # wipe out irrelevant  ?
        y_hat_output_tensor = output_tensor
        y_real_tensor = y_tensor
        weight_real_tensor = weight_tensor
        self.loss = self.loss_layer((y_hat_output_tensor, y_real_tensor, weight_real_tensor))
        self.acc = self.loss_layer.lm_acc
        with tf.control_dependencies([self.save_lstm_state(final_state)]):
            self.loss = tf.identity(self.loss)
        self.train_op = self.optimize(self.loss, self.arguments["max_grad_norm"],
                                      self.arguments["lr"], self.arguments["lr_decay"])
        super(AutoEncoderModel, self).build()

    def fit(self, model_inpus = None, save_model_path = None, pretrained_model_path = None):
        variables_to_restore = self.trainable_weights()
        super(AutoEncoderModel, self)._fit(model_inpus, save_model_path, pretrained_model_path, variables_to_restore=variables_to_restore)

class AdversarialClassificationModel(BaseModel):
    def __init__(self, use_average=False):
        super(AdversarialClassificationModel, self).__init__(use_average=use_average)
        logger.info("constructing classification model dataset...")
        self.inputs, self.get_lstm_state, self.save_lstm_state = construct_classification_model_input_tensor_with_state(
            **self.arguments["adv_cl_inputs"])
        logger.info("classification model dataset is constructed.")
        self.sequences["lm_sequence"] = seq.LanguageModelSequence(**self.arguments["lm_sequence"])
        self.sequences["cl_dense"] = seq.ClassificationModelDenseHeader(
            layer_sizes=[self.arguments["adv_cl_sequence"]["hidden_size"]] * self.arguments["adv_cl_sequence"][
                "num_layers"], input_size=self.arguments["adv_cl_sequence"]["input_size"],
            num_classes=self.arguments["adv_cl_sequence"]["num_classes"],
            keep_prob=self.arguments["adv_cl_sequence"]["keep_prob"])
        self.loss_layer = layers.ClassificationSparseSoftmaxLoss()
        self.train_op = None
        self.adv_loss = None
        self.cl_loss = None
        self.loss = None
        self.acc = None

    def build(self):
        # X_tensor (None, steps)
        # Y_tensor (None,)
        # weight_tensor (None, steps)
        X_tensor, y_tensor, weight_tensor = tf.squeeze(self.inputs.sequences["X"], axis=-1), tf.squeeze(
            self.inputs.context["y"], axis=-1), tf.squeeze(self.inputs.sequences["weight"], axis=-1)
        # note that every sample in a batch has different sequence length, use gather_nd instead of gather
        # laststep_gather_indices [(batch_index, step_index), ...]
        self.laststep_gather_indices = tf.stack([tf.range(self.arguments["inputs"]["batch_size"]), self.inputs.length - 1], 1)
        # [LSTMTuple(c, h), ...]
        lstm_initial_state = self.get_lstm_state()
        # output_tensor (None, steps, lstm_size)
        # final_state (None, lstm_size)
        output_tensor, final_state = self.sequences["lm_sequence"](X_tensor, lstm_initial_state,
                                                                   sequence_len=self.inputs.length)
        # cl_loss ()
        # logits (None, n_classes)
        self.cl_loss, logits = self.get_cl_loss(output_tensor, y_tensor, weight_tensor)
        tf.summary.scalar('classification_loss', self.cl_loss)
        # final_output_weights (None,)
        final_output_weights = tf.gather_nd(weight_tensor, self.laststep_gather_indices)
        self.acc = layers.accuracy(logits, y_tensor, final_output_weights)
        tf.summary.scalar('accuracy', self.acc)
        self.adv_loss = self.get_adversarial_loss(y_tensor, weight_tensor)
        tf.summary.scalar('adversarial_loss', self.adv_loss)
        self.loss = self.cl_loss + self.adv_loss * tf.constant(self.arguments["adv_cl_loss"]["adv_reg_coeff"], name='adv_reg_coeff')
        tf.summary.scalar('classification_loss + adversarial_loss', self.loss)
        with tf.control_dependencies([self.save_lstm_state(final_state)]):
            self.loss = tf.identity(self.loss)
        self.train_op = self.optimize(self.loss, self.arguments["max_grad_norm"],
                                      self.arguments["lr"], self.arguments["lr_decay"])
        super(AdversarialClassificationModel, self).build()

    def fit(self, model_inpus = None, save_model_path = None, pretrained_model_path = None):
        variables_to_restore = self.sequences["lm_sequence"].trainable_weights
        super(AdversarialClassificationModel, self)._fit(model_inpus, save_model_path, pretrained_model_path,
                                        variables_to_restore=variables_to_restore)

    def get_cl_loss(self, lstm_output_tensor, y_tensor, weight_tensor):
        # final_output_tensor (None, lstm_size)
        final_output_tensor = tf.gather_nd(lstm_output_tensor, self.laststep_gather_indices)
        # final_output_weights (None,)
        final_output_weights = tf.gather_nd(weight_tensor, self.laststep_gather_indices)
        # logits (None, n_classes)
        logits = self.sequences["cl_dense"](final_output_tensor)
        # cl_loss ()
        cl_loss = self.loss_layer([logits, y_tensor, final_output_weights])
        return cl_loss, logits

    def get_adversarial_loss(self, y_tensor, weight_tensor):
        # embedding_grads (None, steps, embbed_size)
        embedding_grads = tf.gradients(self.cl_loss, self.sequences["lm_sequence"].embedding)[0]
        embedding_grads = tf.stop_gradient(embedding_grads)
        # pertueb (None, steps, embbed_size)
        perturb = self.embed_scale_l2(embedding_grads, self.arguments["adv_cl_loss"]["perturb_norm_length"])
        # lstm_initial_state  [LSTMTuple(c, h), ...]
        lstm_initial_state = self.get_lstm_state()
        # sequence_len (None,)
        sequence_len = self.inputs.length
        perturbed_embedding = self.sequences["lm_sequence"].embedding + perturb
        # lstm_output_tensor (None, steps, lstm_size)
        lstm_output_tensor, _ = self.sequences["lm_sequence"].lstm_layer(perturbed_embedding, lstm_initial_state, sequence_len)
        return self.get_cl_loss(lstm_output_tensor, y_tensor, weight_tensor)[0]

    def embed_scale_l2(self, x, norm_length):
        # shape(x) = (batch, num_timesteps, d)
        # Divide x by max(abs(x)) for a numerically stable L2 norm.
        # 2norm(x) = a * 2norm(x/a)
        # Scale over the full sequence, dims (1, 2)
        # alpha (None, 1, 1)
        alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
        # l2_norm (None, 1, 1)
        l2_norm = alpha * tf.sqrt(tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
        x_unit = x / l2_norm
        return norm_length * x_unit



