from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
import time
from adversarial_net import arguments as flags
from adversarial_net import sequences as seq
from adversarial_net import layers
from adversarial_net.utils import getLogger
from adversarial_net.inputs import construct_language_model_input_tensor_with_state
from adversarial_net.inputs import construct_classification_model_input_tensor_with_state
from adversarial_net.inputs import construct_autoencoder_model_input_tensor_with_state
from adversarial_net import osp


logger = getLogger("model")
def configure():
    flags.register_variable(name="vocab_freqs")
    flags.add_argument(scope="inputs", name="datapath", argtype=str)
    flags.add_argument(scope="inputs", name="dataset", argtype=str)
    flags.add_argument(scope="inputs", name="batch_size", argtype=int, default=256)
    flags.add_argument(scope="inputs", name="unroll_steps", argtype=int, default=200)
    flags.add_association(scope="inputs", name="lstm_num_layers", assoc_scope="lm_sequence", assoc_name="rnn_num_layers")
    flags.add_association(scope="inputs", name="state_size", assoc_scope="lm_sequence", assoc_name="rnn_cell_size")
    flags.add_argument(scope="inputs", name="bidrec", argtype=bool, default=False)

    flags.add_argument(scope="lm_sequence", name="vocab_size", argtype=int, default=50000)
    flags.add_argument(scope="lm_sequence", name="embedding_dim", argtype=int, default=256)
    flags.add_argument(scope="lm_sequence", name="rnn_cell_size", argtype=int, default=1024)
    flags.add_argument(scope="lm_sequence", name="normalize", argtype="bool", default=True)
    flags.add_argument(scope="lm_sequence", name="keep_embed_prob", argtype=float, default=0.5)
    flags.add_argument(scope="lm_sequence", name="lstm_keep_pro_out", argtype=float, default=1.0)
    flags.add_argument(scope="lm_sequence", name="rnn_num_layers", argtype=int, default=1)
    flags.add_association(scope="lm_sequence", name="vocab_freqs", assoc_name="vocab_freqs")
    flags.add_scope_association(scope="lm_inputs", assoc_scope="inputs")
    flags.add_argument(scope="lm_loss", name="vocab_size", argtype=int, default=50000)
    flags.add_argument(scope="lm_loss", name="num_candidate_samples", argtype=int, default=1024)
    flags.add_association(scope="lm_loss", name="vocab_freqs", assoc_name="vocab_freqs")

    flags.add_scope_association(scope="ae_sequence", assoc_scope="lm_sequence")
    flags.add_scope_association(scope="ae_inputs", assoc_scope="inputs")
    flags.add_scope_association(scope="ae_loss", assoc_scope="lm_loss")

    flags.add_scope_association(scope="adv_cl_inputs", assoc_scope="inputs")
    flags.add_association(scope="adv_cl_inputs", name="phase", assoc_name="phase")
    flags.add_argument(scope="adv_cl_sequence", name="hidden_size", argtype=int, default=30)
    flags.add_argument(scope="adv_cl_sequence", name="num_layers", argtype=int, default=1)
    flags.add_argument(scope="adv_cl_sequence", name="num_classes", argtype=int, default=2)
    flags.add_argument(scope="adv_cl_sequence", name="keep_prob", argtype=int, default=0.5)
    flags.add_association(scope="adv_cl_sequence", name="input_size", assoc_scope="lm_sequence", assoc_name="rnn_cell_size")
    flags.add_argument(scope="adv_cl_loss", name="adv_reg_coeff", argtype=float, default=1.0)
    flags.add_argument(scope="adv_cl_loss", name="perturb_norm_length", argtype=float, default=5.0)

    flags.add_argument(name="phase", argtype=str, default="train")
    flags.add_argument(name="max_grad_norm", argtype=float, default=1.0)
    flags.add_argument(name="lr", argtype=float, default=1e-3)
    flags.add_argument(name="lr_decay", argtype=float, default=0.9999)
    flags.add_argument(name="max_steps", argtype=int, default=100000)
    flags.add_argument(name="save_steps", argtype=int, default=5)
    flags.add_argument(name="save_best", argtype="bool", default=True)
    flags.add_argument(name="save_best_check_steps", argtype=int, default=100)
    flags.add_argument(name="eval_acc", argtype=bool, default=False)
    flags.add_argument(name="eval_steps", argtype=int, default=100)
    flags.add_argument(name="should_restore_if_could", argtype="bool", default=True)
    flags.add_argument(name="tf_debug_trace", argtype=bool, default=False)
configure()


class BaseModel(object):
    def __init__(self, use_average = False):
        self.arguments = flags
        self.sequences = {}
        self.variable_averages = None
        self.use_average = use_average
        self.global_step = tf.train.get_or_create_global_step()
        self.loss_layer = None
        self.built = False
        self.debug_tensors = {}
        self.debug_trace = self.arguments["tf_debug_trace"]

    def build(self):
        self.built = True

    def _fit(self, model_inpus = None, save_model_path = None, pretrained_model_path = None, variables_to_restore = None):
        if not self.built:
            raise Exception("call build() before fitting the model")
        model_phase = 1 if self.arguments["phase"] in ["train"] else 0
        if model_inpus:
            feed_dict = dict(zip(self.inputs, model_inpus))
        else:
            feed_dict = {}
        feed_dict[tf.keras.backend.learning_phase()] = model_phase
        self.run_training(self.train_op, self.loss, acc=self.acc, feed_dict=feed_dict, save_model_path=save_model_path,
                          variables_to_restore=variables_to_restore, pretrained_model_path=pretrained_model_path)

    def trainable_weights(self):
        _trainable_weights = []
        for sequence in self.sequences.values():
            _trainable_weights += sequence.trainable_weights
        if self.loss_layer and isinstance(self.loss_layer, tf.keras.layers.Layer):
            return _trainable_weights + self.loss_layer.trainable_weights
        else:
            return _trainable_weights

    def optimize(self, loss, max_grad_norm, lr, lr_decay):
        with tf.name_scope('optimization'):
            # Compute gradients.
            tvars = tf.trainable_variables()
            grads = tf.gradients(loss, tvars)

            # Clip non-embedding grads
            non_embedding_grads_and_vars = [(g, v) for (g, v) in zip(grads, tvars)
                                            if 'embedding' not in v.op.name]
            embedding_grads_and_vars = [(g, v) for (g, v) in zip(grads, tvars)
                                        if 'embedding' in v.op.name]

            ne_grads, ne_vars = zip(*non_embedding_grads_and_vars)
            ne_grads, _ = tf.clip_by_global_norm(ne_grads, max_grad_norm)
            non_embedding_grads_and_vars = list(zip(ne_grads, ne_vars))

            grads_and_vars = embedding_grads_and_vars + non_embedding_grads_and_vars

            # Decaying learning rate
            lr = tf.train.exponential_decay(lr, self.global_step, 1, lr_decay, staircase=True)
            tf.summary.scalar('learning_rate', lr)
            tf.summary.scalar('loss', loss)
            opt = tf.train.AdamOptimizer(lr)
            if self.use_average:
                self.variable_averages = tf.train.ExponentialMovingAverage(0.999, global_step)
            apply_gradient_op = opt.apply_gradients(grads_and_vars, global_step=self.global_step)
            if self.use_average:
                with tf.control_dependencies([apply_gradient_op]):
                    train_op = self.variable_averages.apply(tvars)
            else:
                train_op = apply_gradient_op
        return train_op

    def maybe_restore_pretrained_model(self, sess, saver_for_restore, model_dir, train_dir):
        """Restores pretrained model if there is no ckpt model."""
        ckpt = tf.train.get_checkpoint_state(train_dir)
        checkpoint_exists = ckpt and ckpt.model_checkpoint_path
        if checkpoint_exists:
            logger.info('Checkpoint exists in train_dir; skipping pretraining restore')
            return
        pretrain_ckpt = tf.train.get_checkpoint_state(model_dir)
        if not (pretrain_ckpt and pretrain_ckpt.model_checkpoint_path):
            logger.info('Asked to restore pretrained sub model from %s but no checkpoint found.' % model_dir)
            return
        logger.info('restore pretrained variables from: %s' % pretrain_ckpt.model_checkpoint_path)
        saver_for_restore.restore(sess, pretrain_ckpt.model_checkpoint_path)

    def run_training(self, train_op, loss, acc=None, feed_dict=None, save_model_path=None, variables_to_restore=None,
                     pretrained_model_path=None):
        saver_for_restore = None
        if pretrained_model_path:
            assert variables_to_restore
            logger.info('Will attempt restore from %s: %s', pretrained_model_path, variables_to_restore)
            saver_for_restore = tf.train.Saver(variables_to_restore)
        model_saver = tf.train.Saver(max_to_keep=1)
        loss_val = best_loss_val = 99999999
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(osp.dirname(save_model_path), sess.graph)
            merged_summary = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())
            if saver_for_restore:
                self.maybe_restore_pretrained_model(sess, saver_for_restore, osp.dirname(pretrained_model_path), osp.dirname(save_model_path))
            if self.arguments["should_restore_if_could"] and save_model_path is not None:
                model_ckpt = tf.train.get_checkpoint_state(osp.dirname(save_model_path))
                model_checkpoint_exists = model_ckpt and model_ckpt.model_checkpoint_path
                if model_checkpoint_exists:
                    logger.info("resotre model from %s" % model_ckpt.model_checkpoint_path)
                    saver_for_model_restore = tf.train.Saver()
                    saver_for_model_restore.restore(sess, model_ckpt.model_checkpoint_path)
            coodinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coodinator)
            global_step_val = 0
            current_steps = sess.run(self.global_step)
            max_steps = self.arguments["max_steps"] + current_steps
            while global_step_val < max_steps:
                start_time = time.time()
                ops = [train_op, loss, self.global_step, merged_summary]
                if acc is not None and self.arguments["eval_acc"]:
                    ops.append(acc)
                else:
                    ops.append(tf.constant(0.0))
                run_options = run_metadata = None
                if self.debug_trace and (global_step_val + 1) % self.arguments["eval_steps"] == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                _, loss_val, global_step_val, summary, acc_val = sess.run(ops, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                if self.debug_tensors:
                    # note that different batch is used when queue is involved in graph
                    debug_results = sess.run(list(self.debug_tensors.values()), feed_dict=feed_dict)
                    debug_results = zip(self.debug_tensors.keys(), debug_results)
                    for key, value in debug_results:
                        logger.info("Debug [%s] eval results: %s" % (key, value))
                duration = time.time() - start_time
                if self.debug_trace and global_step_val % self.arguments["eval_steps"] == 0:
                    summary_writer.add_run_metadata(run_metadata, "step-%d" % global_step_val)
                summary_writer.add_summary(summary, global_step_val)
                # Logging
                if global_step_val % self.arguments["eval_steps"] == 0:
                    if not self.arguments["eval_acc"]:
                        logger.info("loss at step-%s/%s: %s, duration: %s" % (global_step_val, max_steps, loss_val, duration))
                    elif acc is not None and self.arguments["eval_acc"]:
                        logger.info(
                            "loss at step-%s/%s: %s, acc: %s, duration: %s" % (global_step_val, max_steps, loss_val, acc_val, duration))
                if save_model_path is not None:
                    # save best
                    if self.arguments["save_best"] and loss_val < best_loss_val and global_step_val % self.arguments["save_best_check_steps"] == 0:
                        logger.info("save best to {}".format(save_model_path))
                        model_saver.save(sess, save_model_path, self.global_step)
                    # save model per save_steps
                    if not self.arguments["save_best"] and global_step_val % self.arguments["save_steps"] == 0:
                        logger.info("save model.")
                        model_saver.save(sess, save_model_path, self.global_step)
            coodinator.request_stop()
            coodinator.join(threads)
            if save_model_path is not None:
                if not self.arguments["save_best"]:
                    logger.info("save model.")
                    model_saver.save(sess, save_model_path, self.global_step)
                else:
                    if global_step_val % self.arguments["save_best_check_steps"] != 0 and loss_val < best_loss_val:
                        logger.info("save model.")
                        model_saver.save(sess, save_model_path, self.global_step)


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



