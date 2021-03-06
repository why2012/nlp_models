from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python import debug as tf_debug
import time
from adversarial_net import arguments as flags
from adversarial_net.utils import getLogger
from adversarial_net import osp
from collections import defaultdict

# batch_size 256
# unroll_steps 400
# rnn_cell_size 1024
# vocab_size 86934
def configure():
    flags.register_variable(name="vocab_freqs")
    flags.add_argument(scope="inputs", name="datapath", argtype=str)
    flags.add_argument(scope="inputs", name="dataset", argtype=str)
    flags.add_argument(scope="inputs", name="eval_count_examples", argtype=int, default=-1)
    flags.add_argument(scope="inputs", name="vocab_size", argtype=int, default=50000)
    flags.add_association(scope="inputs", name="eval_max_words", assoc_scope="inputs", assoc_name="vocab_size")
    flags.add_argument(scope="inputs", name="batch_size", argtype=int, default=256)
    flags.add_argument(scope="inputs", name="unroll_steps", argtype=int, default=200)
    flags.add_association(scope="inputs", name="lstm_num_layers", assoc_scope="lm_sequence", assoc_name="rnn_num_layers")
    flags.add_association(scope="inputs", name="state_size", assoc_scope="lm_sequence", assoc_name="rnn_cell_size")
    flags.add_argument(scope="inputs", name="bidrec", argtype=bool, default=False)

    flags.add_association(scope="lm_sequence", name="vocab_size", assoc_scope="inputs", assoc_name="vocab_size")
    flags.add_argument(scope="lm_sequence", name="embedding_dim", argtype=int, default=256)
    # should be same with gan rnn_cell_size
    flags.add_argument(scope="lm_sequence", name="rnn_cell_size", argtype=int, default=1024)
    flags.add_argument(scope="lm_sequence", name="normalize", argtype="bool", default=True)
    flags.add_argument(scope="lm_sequence", name="keep_embed_prob", argtype=float, default=1.0)
    flags.add_argument(scope="lm_sequence", name="lstm_keep_pro_out", argtype=float, default=1.0)
    flags.add_argument(scope="lm_sequence", name="rnn_num_layers", argtype=int, default=1)
    flags.add_association(scope="lm_sequence", name="vocab_freqs", assoc_name="vocab_freqs")
    flags.add_scope_association(scope="lm_inputs", assoc_scope="inputs")
    flags.add_association(scope="lm_loss", name="vocab_size", assoc_scope="inputs", assoc_name="vocab_size")
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
    flags.add_argument(scope="adv_cl_sequence", name="keep_prob", argtype=int, default=1.0)
    flags.add_association(scope="adv_cl_sequence", name="input_size", assoc_scope="lm_sequence", assoc_name="rnn_cell_size")
    flags.add_argument(scope="adv_cl_loss", name="adv_reg_coeff", argtype=float, default=1.0)
    flags.add_argument(scope="adv_cl_loss", name="perturb_norm_length", argtype=float, default=5.0)

    flags.add_argument(scope="vir_adv_loss", name="num_power_iteration", argtype=int, default=1)
    flags.add_argument(scope="vir_adv_loss", name="small_constant_for_finite_diff", argtype=float, default=1e-1)

    flags.add_argument(scope="gan", name="critic_iters", argtype=int, default=5)
    flags.add_argument(scope="gan", name="rnn_cell_size", argtype=int, default=1024)

    flags.add_argument(scope="summary", name="rnn_cell_size", argtype=int, default=512)
    flags.add_argument(scope="summary", name="rnn_keep_prob_out", argtype=float, default=1.0)
    flags.add_argument(scope="summary", name="maximum_iterations", argtype=int, default=50)
    flags.add_argument(scope="summary", name="beam_width", argtype=int, default=3)

    flags.add_argument(name="phase", argtype=str, default="train")
    flags.add_argument(name="max_grad_norm", argtype=float, default=1.0)
    flags.add_argument(name="lr", argtype=float, default=1e-3)
    flags.add_argument(name="lr_decay", argtype=float, default=0.9999)
    flags.add_argument(name="max_steps", argtype=int, default=100000)
    flags.add_argument(name="save_steps", argtype=int, default=100)
    flags.add_argument(name="save_best", argtype="bool", default=True)
    flags.add_argument(name="save_best_check_steps", argtype=int, default=100)
    flags.add_argument(name="eval_acc", argtype=bool, default=False)
    flags.add_argument(name="eval_steps", argtype=int, default=100)
    flags.add_argument(name="should_restore_if_could", argtype="bool", default=True)
    flags.add_argument(name="tf_debug_trace", argtype=bool, default=False)
    flags.add_argument(name="tf_timeline_dir", argtype=str, default=None)
    flags.add_argument(name="no_need_clip_grads", argtype=bool, default=False)
    flags.add_argument(name="with_lr_decay", argtype="bool", default=True)
    flags.add_argument(name="lr_decay_step", argtype=int, default=1)
    # continue from break point
    flags.add_argument(name="best_loss_val", argtype=float, default=99999999.0)
    flags.add_argument(name="extra_save_dir", argtype=str, default=None)
    flags.add_argument(name="tfdebug_root", argtype=str, default=None)
    flags.add_argument(name="use_exp_mov_avg_loss", argtype=bool, default=False)
    flags.add_argument(name="use_exp_mov_avg_loss_decay", argtype=float, default=0.9)

    flags.add_argument(name="logging_file", argtype=str, default=None)
configure()
logger = getLogger("model")

class VariableManager(object):
    def __init__(self):
        self.collections = defaultdict(list)

    def add_to_collection(self, name, value):
        self.collections[name].append(value)

    def get_collection(self, name):
        return self.collections[name]

class BaseModel(object):
    CLIP_GRADS_SCOPE = "clip_grads_scope"

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
        self.var_manager = VariableManager()
        self.extra_save_path = self.arguments["extra_save_dir"]
        self.acc = None
        self.exp_mov_avg_loss = None

    def _get_save_path(self, save_model_path):
        original_save_model_path, filepath = save_model_path.rsplit("/", 1)
        if self.extra_save_path:
            extra_save_model_path = osp.join(original_save_model_path, self.extra_save_path)
            extra_save_model_path = osp.join(extra_save_model_path, filepath)
        else:
            extra_save_model_path = None
        original_save_model_path = osp.join(original_save_model_path, filepath)
        if self.arguments["save_best"]:
            save_best_path = original_save_model_path
            save_steps_path = extra_save_model_path
        else:
            save_steps_path = original_save_model_path
            save_best_path = extra_save_model_path
        return save_best_path, save_steps_path

    def build(self):
        self.built = True

    def _fit(self, model_inpus = None, save_model_path = None, pretrained_model_path = None, variables_to_restore = None, sess = None, debug_mode = False):
        if not self.built:
            raise Exception("call build() before fitting the model")
        model_phase = 1 if self.arguments["phase"] in ["train"] else 0
        if model_inpus:
            feed_dict = dict(zip(self.inputs, model_inpus))
        else:
            feed_dict = {}
        feed_dict[tf.keras.backend.learning_phase()] = model_phase
        self.run_training(self.train_op, self.loss, acc=self.acc, feed_dict=feed_dict, save_model_path=save_model_path,
                          variables_to_restore=variables_to_restore, pretrained_model_path=pretrained_model_path, sess=sess, debug_mode=debug_mode)

    def trainable_weights(self):
        _trainable_weights = []
        for sequence in self.sequences.values():
            _trainable_weights += sequence.trainable_weights
        if self.loss_layer and isinstance(self.loss_layer, tf.keras.layers.Layer):
            return _trainable_weights + self.loss_layer.trainable_weights
        else:
            return _trainable_weights

    def _get_and_clip_grads_by_variables(self, loss, variables, max_grad_norm, exclude_op_names=[], no_need_clip_grads=None):
        if no_need_clip_grads is None:
            no_need_clip_grads = self.arguments["no_need_clip_grads"]
        def in_exclude_op_names(op_name):
            for exclude_name in exclude_op_names:
                if exclude_name in op_name:
                    return True
            return False
        exclude_vars = list(filter(lambda x: in_exclude_op_names(x.op.name), variables))
        need_clip_vars = list(filter(lambda x: not in_exclude_op_names(x.op.name), variables))
        grads_and_vars = []
        if exclude_vars:
            grads = tf.gradients(loss, exclude_vars)
            grads_and_vars.extend(list(zip(grads, exclude_vars)))
        if need_clip_vars:
            grads = tf.gradients(loss, need_clip_vars)
            if no_need_clip_grads:
                clipped_grads = grads
            else:
                clipped_grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
            grads_and_vars.extend(list(zip(clipped_grads, need_clip_vars)))
        return grads_and_vars

    def _trainable_variables_filter(self, filter_func, vars_list = None):
        return list(filter(filter_func, tf.trainable_variables() if vars_list is None else vars_list))

    def _trainable_variables_filter_and_grads(self, loss, filter_func, vars_list = None):
        tvars = list(filter(filter_func, tf.trainable_variables() if vars_list is None else vars_list))
        grads = tf.gradients(loss, tvars)
        return list(zip(grads, tvars))

    def _get_train_op_with_lr_decay(self, grads_and_vars, global_step, lr=None, lr_decay=None, staircase=True,
                                    optimizer=tf.train.AdamOptimizer, optimizer_kwargs = {}):
        lr = self.arguments["lr"] if lr is None else lr
        lr_decay = self.arguments["lr_decay"] if lr_decay is None else lr_decay
        if self.arguments["with_lr_decay"]:
            lr = tf.train.exponential_decay(lr, global_step, self.arguments["lr_decay_step"], lr_decay, staircase=staircase)
        opt = optimizer(lr, **optimizer_kwargs)
        apply_gradient_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
        return apply_gradient_op, lr

    def _moving_average_wrapper(self, train_op, tvars, global_step, decay = 0.999):
        variable_averages = tf.train.ExponentialMovingAverage(decay, global_step)
        with tf.control_dependencies([train_op]):
            train_op = variable_averages.apply(tvars)
        return train_op

    def exponential_moving_average_loss(self, train_op, loss, global_step, decay = -1):
        if self.arguments["use_exp_mov_avg_loss"]:
            if decay == -1:
                decay = self.arguments["use_exp_mov_avg_loss_decay"]
            variable_averages = tf.train.ExponentialMovingAverage(decay, global_step)
            with tf.control_dependencies([train_op]):
                train_op = variable_averages.apply([loss])
            exp_mov_avg_loss = variable_averages.average(loss)
            tf.summary.scalar('exp_mov_avg_loss', exp_mov_avg_loss)
            return train_op, exp_mov_avg_loss
        else:
            return train_op, loss

    def optimize(self, loss, max_grad_norm, lr, lr_decay, lock_embedding=False, norm_embedding=False,
                 optimizer=tf.train.AdamOptimizer, optimizer_kwargs={}):
        with tf.name_scope('optimization'):
            if lock_embedding:
                embedding_grads_and_vars = []
            else:
                embedding_grads_and_vars = self._trainable_variables_filter_and_grads(loss, lambda v: "embedding" in v.op.name)
                if norm_embedding:
                    embedding_grads = list(map(lambda x: x[0], embedding_grads_and_vars))
                    embedding_vars = list(map(lambda x: x[1], embedding_grads_and_vars))
                    embedding_clipped_grads, _ = tf.clip_by_global_norm(embedding_grads, max_grad_norm)
                    embedding_grads_and_vars = list(zip(embedding_clipped_grads, embedding_vars))
            non_embedding_grads_and_vars = self._get_and_clip_grads_by_variables(loss, self._trainable_variables_filter(
                lambda v: "embedding" not in v.op.name), max_grad_norm)
            grads_and_vars = embedding_grads_and_vars + non_embedding_grads_and_vars
            # Decaying learning rate
            train_op, lr = self._get_train_op_with_lr_decay(grads_and_vars, self.global_step, lr=lr, lr_decay=lr_decay,
                                                            optimizer=optimizer, optimizer_kwargs=optimizer_kwargs)
            train_op, exp_mov_avg_loss = self.exponential_moving_average_loss(train_op, loss, self.global_step)
            self.exp_mov_avg_loss = exp_mov_avg_loss
            tf.summary.scalar('learning_rate', lr)
            tf.summary.scalar('loss', loss)
            if self.use_average:
                train_op = self._moving_average_wrapper(train_op, tf.trainable_variables(), self.global_step)
        return train_op

    def _maybe_restore_pretrained_model(self, sess, saver_for_restore, model_dir, train_dir):
        """Restores pretrained model if there is no ckpt model."""
        if train_dir:
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

    def _restore_pretained_variables(self, sess, pretrained_model_path, variables_to_restore, save_model_path = None, saver_for_restore = None):
        if pretrained_model_path:
            if variables_to_restore is None and saver_for_restore:
                variables_to_restore = saver_for_restore._var_list
            logger.info('Will attempt restore from %s: %s', pretrained_model_path, variables_to_restore)
            if saver_for_restore is None:
                assert variables_to_restore
                saver_for_restore = tf.train.Saver(variables_to_restore)
            self._maybe_restore_pretrained_model(sess, saver_for_restore, osp.dirname(pretrained_model_path), osp.dirname(save_model_path))

    def _resotre_training_model(self, sess, save_model_path, saver_for_model_restore = None):
        if self.arguments["should_restore_if_could"] and save_model_path is not None:
            model_ckpt = tf.train.get_checkpoint_state(osp.dirname(save_model_path))
            model_checkpoint_exists = model_ckpt and model_ckpt.model_checkpoint_path
            if model_checkpoint_exists:
                self.check_best_loss_val()
                logger.info("resotre model from %s" % model_ckpt.model_checkpoint_path)
                if saver_for_model_restore is None:
                    # vars = []
                    # for var in tf.global_variables():
                    #     if "ExponentialMovingAverage" not in var.op.name:
                    #         vars.append(var)
                    #     else:
                    #         print("----------", var)
                    # saver_for_model_restore = tf.train.Saver(vars)
                    saver_for_model_restore = tf.train.Saver()
                saver_for_model_restore.restore(sess, model_ckpt.model_checkpoint_path)

    def check_best_loss_val(self):
        if self.arguments["best_loss_val"] == 99999999.0:
            continue_or_num = input("trained model exists, but no specific best_loss_val is defined, continue (Y/N/best_loss_val)[input Y when test]: ").strip().upper()
            if continue_or_num == "Y":
                return
            elif continue_or_num.isdigit():
                self.arguments.set_real_argument_value(name="best_loss_val", value=float(continue_or_num))
            else:
                exit(0)

    def _pretrain_step(self, global_step_val):
        run_options = run_metadata = None
        if self.debug_trace and (global_step_val + 1) % self.arguments["eval_steps"] == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        return run_options, run_metadata

    def _train_step(self, sess, ops, acc = None, feed_dict = None, run_options = None, run_metadata = None):
        # training phase
        if acc is not None and self.arguments["eval_acc"]:
            ops.append(acc)
            return_cache = sess.run(ops, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
        else:
            acc_val = 0.
            return_cache = sess.run(ops, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            return_cache.append(acc_val)
        return return_cache[1:]

    def _summary_step(self, sess, debug_tensors, global_step_val, summary_writer, summary, run_metadata = None, feed_dict = None):
        if debug_tensors:
            # note that different batch is used when queue is involved in graph
            debug_results = sess.run(list(debug_tensors.values()), feed_dict=feed_dict)
            debug_results = zip(debug_tensors.keys(), debug_results)
            for key, value in debug_results:
                logger.info("Debug [%s] eval results: %s" % (key, value))
        if self.debug_trace and global_step_val % self.arguments["eval_steps"] == 0 and run_metadata:
            if summary_writer is not None:
                if isinstance(run_metadata, list):
                    for i, metadata in enumerate(run_metadata):
                        summary_writer.add_run_metadata(metadata, "%d-step-%d" % (i, global_step_val))
                else:
                    summary_writer.add_run_metadata(run_metadata, "step-%d" % global_step_val)
            if self.timeline_dir:
                if isinstance(run_metadata, list):
                    for i, metadata in enumerate(run_metadata):
                        fetched_timeline = timeline.Timeline(metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        with open(osp.join(self.timeline_dir, '%s_timeline_id_%d_step_%d.json' % (self.model_name, i, global_step_val)), 'w') as f:
                            f.write(chrome_trace)
                else:
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open(osp.join(self.timeline_dir, '%s_timeline_step_%d.json' % (self.model_name, global_step_val)), 'w') as f:
                        f.write(chrome_trace)
        if summary_writer is not None and summary:
            if isinstance(summary, list):
                for summary_item in summary:
                    summary_writer.add_summary(summary_item, global_step_val)
            else:
                summary_writer.add_summary(summary, global_step_val)

    def _eval_step(self, global_step_val, max_steps, loss_val, acc_val = -1, duration = -1):
        if global_step_val % self.arguments["eval_steps"] == 0:
            if not self.arguments["eval_acc"]:
                logger.info("loss at step-%s/%s: %s, duration: %s" % (global_step_val, max_steps, loss_val, duration))
            elif (isinstance(acc_val, dict) or acc_val > 0) and self.arguments["eval_acc"]:
                logger.info(
                    "loss at step-%s/%s: %s, acc: %s, duration: %s" % (
                    global_step_val, max_steps, loss_val, acc_val, duration))

    def _save_model_step(self, sess, model_saver, save_model_path, loss_val, best_loss_val, global_step_val):
        if save_model_path is not None:
            save_best_path, save_steps_path = self._get_save_path(save_model_path)
            # save best
            if (self.arguments["save_best"] or self.extra_save_path) and loss_val < best_loss_val and global_step_val % self.arguments[
                "save_best_check_steps"] == 0:
                logger.info("save best to {}".format(save_best_path))
                model_saver.save(sess, save_best_path, global_step_val)
                best_loss_val = loss_val
                with open(osp.join(osp.dirname(save_best_path), "best_loss_records.txt"), "a+") as recordf:
                    recordf.write("step-loss: %s - %s\n" % (global_step_val, best_loss_val))
            # save model per save_steps
            if (not self.arguments["save_best"] or self.extra_save_path) and global_step_val % self.arguments["save_steps"] == 0:
                logger.info("save model to {}".format(save_steps_path))
                model_saver.save(sess, save_steps_path, global_step_val)
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    with open(osp.join(osp.dirname(save_steps_path), "best_loss_records.txt"), "a+") as recordf:
                        recordf.write("step-loss: %s - %s\n" % (global_step_val, best_loss_val))
        return best_loss_val

    def _initialize_process(self, sess, save_model_path):
        model_saver = tf.train.Saver(max_to_keep=1)
        summary_writer = None
        if save_model_path is not None:
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
            save_best_path, save_steps_path = self._get_save_path(save_model_path)
            if not self.arguments["save_best"] or self.extra_save_path:
                logger.info("save model.")
                model_saver.save(sess, save_steps_path, global_step_val)
                with open(osp.join(osp.dirname(save_steps_path), "best_loss_records.txt"), "a+") as recordf:
                    recordf.write("step-loss: %s - %s\n" % (global_step_val, loss_val))
            if self.arguments["save_best"] or self.extra_save_path:
                if global_step_val % self.arguments["save_best_check_steps"] != 0 and loss_val < best_loss_val:
                    logger.info("save model.")
                    model_saver.save(sess, save_best_path, global_step_val)
                    with open(osp.join(osp.dirname(save_best_path), "best_loss_records.txt"), "a+") as recordf:
                        recordf.write("step-loss: %s - %s\n" % (global_step_val, loss_val))

    def make_restore_average_vars_dict(self, variables):
        var_restore_dict = {}
        variable_averages = tf.train.ExponentialMovingAverage(0.999)
        for v in variables:
            if v in tf.trainable_variables():
                name = variable_averages.average_name(v)
            else:
                name = v.op.name
            var_restore_dict[name] = v
        return var_restore_dict

    def run_training(self, train_op, loss, acc=None, feed_dict=None, save_model_path=None, variables_to_restore=None,
                     pretrained_model_path=None, sess = None, debug_mode = False):
        loss_val = best_loss_val = self.arguments["best_loss_val"]
        global_step_val = 0
        if sess is None:
            sess = tf.Session()
        if debug_mode:
            if save_model_path:
                dump_root = osp.join(save_model_path, "dump")
            else:
                dump_root = self.arguments["tfdebug_root"]
            sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root=dump_root)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        with sess:
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
                if self.arguments["use_exp_mov_avg_loss"]:
                    ops = [train_op, loss, self.global_step, merged_summary, self.exp_mov_avg_loss]
                else:
                    ops = [train_op, loss, self.global_step, merged_summary]
                return_cache = self._train_step(sess=sess, ops=ops, acc=acc, feed_dict=feed_dict, run_options=run_options,
                                                run_metadata=run_metadata)
                if self.arguments["use_exp_mov_avg_loss"]:
                    loss_val, global_step_val, summary, exp_mov_avg_loss_val, acc_val = return_cache
                    loss_val = exp_mov_avg_loss_val
                else:
                    loss_val, global_step_val, summary, acc_val = return_cache
                duration = time.time() - start_time
                # summary & debug trace phase
                self._summary_step(sess=sess, debug_tensors=self.debug_tensors, global_step_val=global_step_val,
                                   summary_writer=summary_writer, summary=summary, run_metadata=run_metadata, feed_dict=feed_dict)
                # Logging
                self._eval_step(global_step_val, max_steps, loss_val, acc_val, duration)
                # save model if could
                best_loss_val = self._save_model_step(sess, model_saver, save_model_path, loss_val, best_loss_val, global_step_val)
            self._finish_process(sess, coodinator, threads, model_saver, save_model_path, global_step_val, loss_val, best_loss_val)
