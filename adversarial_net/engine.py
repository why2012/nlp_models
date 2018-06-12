from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.client import timeline
import time
from adversarial_net import arguments as flags
from adversarial_net.utils import getLogger
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

    flags.add_argument(scope="vir_adv_loss", name="num_power_iteration", argtype=int, default=1)
    flags.add_argument(scope="vir_adv_loss", name="small_constant_for_finite_diff", argtype=float, default=1e-1)

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
    flags.add_argument(name="tf_timeline_dir", argtype=str, default=None)
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
        self.timeline_dir = self.arguments["tf_timeline_dir"]
        self.model_name = self.__class__.__name__

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

    def _restore_pretained_variables(self, sess, pretrained_model_path, save_model_path, variables_to_restore):
        if pretrained_model_path:
            assert variables_to_restore
            logger.info('Will attempt restore from %s: %s', pretrained_model_path, variables_to_restore)
            saver_for_restore = tf.train.Saver(variables_to_restore)
            self.maybe_restore_pretrained_model(sess, saver_for_restore, osp.dirname(pretrained_model_path), osp.dirname(save_model_path))

    def _resotre_training_model(self, sess, save_model_path):
        if self.arguments["should_restore_if_could"] and save_model_path is not None:
            model_ckpt = tf.train.get_checkpoint_state(osp.dirname(save_model_path))
            model_checkpoint_exists = model_ckpt and model_ckpt.model_checkpoint_path
            if model_checkpoint_exists:
                logger.info("resotre model from %s" % model_ckpt.model_checkpoint_path)
                saver_for_model_restore = tf.train.Saver()
                saver_for_model_restore.restore(sess, model_ckpt.model_checkpoint_path)

    def _pretrain_step(self, global_step_val):
        run_options = run_metadata = None
        if self.debug_trace and (global_step_val + 1) % self.arguments["eval_steps"] == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        return run_options, run_metadata

    def _train_step(self, sess, ops, acc, feed_dict = None, run_options = None, run_metadata = None):
        # training phase
        if acc is not None and self.arguments["eval_acc"]:
            ops.append(acc)
            _, loss_val, global_step_val, summary, acc_val = sess.run(ops, feed_dict=feed_dict,
                                                                      options=run_options,
                                                                      run_metadata=run_metadata)
        else:
            acc_val = 0.
            _, loss_val, global_step_val, summary = sess.run(ops, feed_dict=feed_dict,
                                                             options=run_options,
                                                             run_metadata=run_metadata)
        return loss_val, global_step_val, summary, acc_val

    def _summary_step(self, debug_tensors, global_step_val, summary_writer, summary, run_metadata = None, feed_dict = None):
        if debug_tensors:
            # note that different batch is used when queue is involved in graph
            debug_results = sess.run(list(debug_tensors.values()), feed_dict=feed_dict)
            debug_results = zip(debug_tensors.keys(), debug_results)
            for key, value in debug_results:
                logger.info("Debug [%s] eval results: %s" % (key, value))
        if self.debug_trace and global_step_val % self.arguments["eval_steps"] == 0 and run_metadata:
            summary_writer.add_run_metadata(run_metadata, "step-%d" % global_step_val)
            if self.timeline_dir:
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open(osp.join(self.timeline_dir, '%s_timeline_step_%d.json' % (self.model_name, global_step_val)), 'w') as f:
                    f.write(chrome_trace)
        summary_writer.add_summary(summary, global_step_val)

    def _eval_step(self, global_step_val, max_steps, loss_val, acc_val, duration):
        if global_step_val % self.arguments["eval_steps"] == 0:
            if not self.arguments["eval_acc"]:
                logger.info("loss at step-%s/%s: %s, duration: %s" % (global_step_val, max_steps, loss_val, duration))
            elif acc_val > 0 and self.arguments["eval_acc"]:
                logger.info(
                    "loss at step-%s/%s: %s, acc: %s, duration: %s" % (
                    global_step_val, max_steps, loss_val, acc_val, duration))

    def _save_model_step(self, sess, model_saver, save_model_path, loss_val, best_loss_val, global_step_val):
        if save_model_path is not None:
            # save best
            if self.arguments["save_best"] and loss_val < best_loss_val and global_step_val % self.arguments[
                "save_best_check_steps"] == 0:
                logger.info("save best to {}".format(save_model_path))
                model_saver.save(sess, save_model_path, global_step_val)
            # save model per save_steps
            if not self.arguments["save_best"] and global_step_val % self.arguments["save_steps"] == 0:
                logger.info("save model.")
                model_saver.save(sess, save_model_path, global_step_val)

    def _initialize_process(self, sess, save_model_path):
        model_saver = tf.train.Saver(max_to_keep=1)
        summary_writer = tf.summary.FileWriter(osp.dirname(save_model_path), sess.graph)
        merged_summary = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        current_steps = sess.run(self.global_step)
        coodinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coodinator)

        return model_saver, summary_writer, merged_summary, coodinator, threads, current_steps

    def _finish_process(self, sess, coodinator, threads, model_saver, save_model_path, global_step_val, loss_val, best_loss_val):
        coodinator.request_stop()
        coodinator.join(threads)
        if save_model_path is not None:
            if not self.arguments["save_best"]:
                logger.info("save model.")
                model_saver.save(sess, save_model_path, global_step_val)
            else:
                if global_step_val % self.arguments["save_best_check_steps"] != 0 and loss_val < best_loss_val:
                    logger.info("save model.")
                    model_saver.save(sess, save_model_path, global_step_val)

    def run_training(self, train_op, loss, acc=None, feed_dict=None, save_model_path=None, variables_to_restore=None,
                     pretrained_model_path=None):
        loss_val = best_loss_val = 99999999
        global_step_val = 0
        with tf.Session() as sess:
            model_saver, summary_writer, merged_summary, coodinator, threads, current_steps = self._initialize_process(sess, save_model_path)
            # pretained model restore step
            self._restore_pretained_variables(sess=sess, pretrained_model_path=pretrained_model_path,
                                              save_model_path=save_model_path,
                                              variables_to_restore=variables_to_restore)
            # store model step
            self._resotre_training_model(sess=sess, save_model_path=save_model_path)
            max_steps = self.arguments["max_steps"] + current_steps
            while global_step_val < max_steps:
                # pre-train phase
                run_options, run_metadata = self._pretrain_step(global_step_val)
                start_time = time.time()
                # train step
                ops = [train_op, loss, self.global_step, merged_summary]
                loss_val, global_step_val, summary, acc_val = self._train_step(sess=sess, ops=ops, acc=acc,
                                                                               feed_dict=feed_dict, run_options=run_options,
                                                                               run_metadata=run_metadata)
                duration = time.time() - start_time
                # summary & debug trace phase
                self._summary_step(debug_tensors=self.debug_tensors, global_step_val=global_step_val,
                                   summary_writer=summary_writer, summary=summary, run_metadata=run_metadata, feed_dict=feed_dict)
                # Logging
                self._eval_step(global_step_val, max_steps, loss_val, acc_val, duration)
                # save model if could
                self._save_model_step(sess, model_saver, save_model_path, loss_val, best_loss_val, global_step_val)
            self._finish_process(sess, coodinator, threads, model_saver, save_model_path, global_step_val, loss_val, best_loss_val)
