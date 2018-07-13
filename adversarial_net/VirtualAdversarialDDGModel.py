from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
from adversarial_net.engine import BaseModel
from adversarial_net import arguments as flags
from adversarial_net import sequences as seq
from adversarial_net import layers
from adversarial_net.utils import getLogger
from adversarial_net import osp
from adversarial_net.AdversarialDDGModel import AdversarialDDGModel, modules

logger = getLogger("model")
EOS_TAG = 2

class VirtualAdversarialDDGModel(AdversarialDDGModel):
    def __init__(self, use_average=False, init_modules=modules.keys()):
        super(VirtualAdversarialDDGModel, self).__init__(use_average=use_average, init_modules=init_modules)
        modules_abbreviation = init_modules
        # ADV_LOSS
        if "ADV_LOSS" in modules_abbreviation:
            self.adversarial_loss = seq.VirtualAdversarialLoss(
                perturb_norm_length=self.arguments["adv_cl_loss"]["perturb_norm_length"],
                small_constant_for_finite_diff=self.arguments["vir_adv_loss"]["small_constant_for_finite_diff"],
                iter_count=self.arguments["vir_adv_loss"]["num_power_iteration"])

    def compute_adv_loss(self, logits, cl_loss, embedding, y_tensor, weight_tensor, sequence_len, laststep_gather_indices, get_lstm_state_fn, seq2seq_fn, dense_fn, eos_indicators):
        def local_compute_logits(perturbed_embedding):
            cl_loss, final_states, cl_logits = self.compute_cl_loss(perturbed_embedding, y_tensor, weight_tensor, sequence_len, laststep_gather_indices, get_lstm_state_fn, seq2seq_fn, dense_fn, return_logits=True)
            return cl_logits
        adv_loss = self.adversarial_loss(compute_logits_fn=local_compute_logits, logits=logits, target=embedding, eos_indicators=eos_indicators, sequence_length=sequence_len)
        return adv_loss * tf.constant(self.arguments["adv_cl_loss"]["adv_reg_coeff"], name='adv_reg_coeff')