import sys
sys.path.insert(0, ".")
from adversarial_net.models import LanguageModel, AutoEncoderModel
from adversarial_net.AdversarialDDGModel import AdversarialDDGModel
from adversarial_net import arguments as flags
from adversarial_net.preprocessing import WordCounter
from adversarial_net import osp
training_step_vals = ["train_lm_model", "pretrain_cl_model", "train_ae_model", "train_generator", "train_topic_generator",
                      "train_cl_model", "eval_generator", "eval_cl_model"]
model_save_suffixes = {
    "train_lm_model": "lm_model/lm_model.ckpt",
    "pre_train_cl_model": "adv_cl_model/adv_cl_model.ckpt",
    "train_ae_model": "ae_model/ae_model.ckpt",
    "train_generator": "generator/generator.ckpt",
    "train_topic_generator": "topic_generator/topic_generator.ckpt",
    "train_cl_model": "final_cl_model/final_cl_model.ckpt"
}
def training_step(value):
    assert value in training_step_vals, "step is one of %s" % training_step_vals
    return value
def eval_from(value):
    eval_from_vals = ["generator", "topic_generator", "pretrain_cl", "final_cl"]
    assert value in eval_from_vals, "step is one of %s" % eval_from_vals
    return value
flags.add_argument(name="step", argtype=training_step)
flags.add_argument(name="save_model_dir", argtype=str)
flags.add_argument(name="pretrain_model_dir", argtype=str, default=None)
flags.add_argument(name="eval_from", argtype=eval_from, default="generator")
flags.add_argument(name="eval_batch_size", argtype=int, default=2)
flags.add_argument(name="eval_topic_count", argtype=int, default=2)
flags.add_argument(name="eval_seq_length", argtype=int, default=200)

# training process         (->embed)
#                  |--> training lm_model |         (->embed)                 (lock embed)              (lock embed)               (->embed)
# start training --|       (lock embed)   |--> pre-training cl_model --> | training generator --> training topic generator --> | re-train cl_model
#                  |--> training ae_model |                              -> eval generator                                     -> eval cl model

def train_lm_model(model_save_suffix = model_save_suffixes["train_lm_model"]):
    save_model_path = osp.join(flags.save_model_dir, model_save_suffix)
    lm_model = LanguageModel()
    lm_model.build()
    lm_model.fit(save_model_path=save_model_path)

def pre_train_cl_model(model_save_suffix = model_save_suffixes["pre_train_cl_model"]):
    assert flags.pretrain_model_dir, "pretrain_model_dir is required"
    save_model_path = osp.join(flags.save_model_dir, model_save_suffix)
    pretrained_model_pathes = {
        "EMBEDDING": osp.join(flags.pretrain_model_dir, model_save_suffixes["train_lm_model"]),
        "T_S": osp.join(flags.pretrain_model_dir, model_save_suffixes["train_lm_model"])
    }
    adv_cl_model = AdversarialDDGModel(init_modules=AdversarialDDGModel.stepB_modules)
    adv_cl_model.build(stepB=True, restorer_tag_notifier=[])
    adv_cl_model.fit(save_model_path=save_model_path, pretrain_model_pathes=pretrained_model_pathes)

def train_ae_model(model_save_suffix=model_save_suffixes["train_ae_model"]):
    assert flags.pretrain_model_dir, "pretrain_model_dir is required"
    save_model_path = osp.join(flags.save_model_dir, model_save_suffix)
    pretrained_model_path = osp.join(flags.pretrain_model_dir, model_save_suffixes["train_lm_model"])
    ae_model = AutoEncoderModel(lock_embedding=True)
    ae_model.build()
    ae_model.fit(save_model_path=save_model_path, pretrained_model_path=pretrained_model_path)

def train_generator(model_save_suffix=model_save_suffixes["train_generator"]):
    assert flags.pretrain_model_dir, "pretrain_model_dir is required"
    save_model_path = osp.join(flags.save_model_dir, model_save_suffix)
    pretrained_model_pathes = {
        "EMBEDDING": osp.join(flags.pretrain_model_dir, model_save_suffixes["pre_train_cl_model"]),
        "FG_S": osp.join(flags.pretrain_model_dir, model_save_suffixes["train_lm_model"]),
        "SEQ_G_LSTM_1": osp.join(flags.pretrain_model_dir, model_save_suffixes["train_lm_model"]),
        "SEQ_G_LSTM_2": osp.join(flags.pretrain_model_dir, model_save_suffixes["train_ae_model"]),
    }
    generator_model = AdversarialDDGModel(init_modules=AdversarialDDGModel.stepA_modules)
    generator_model.build(stepA=True, restorer_tag_notifier=["EMBEDDING"])
    generator_model.fit(save_model_path=save_model_path, pretrain_model_pathes=pretrained_model_pathes)

def train_topic_generator(model_save_suffix=model_save_suffixes["train_topic_generator"]):
    assert flags.pretrain_model_dir, "pretrain_model_dir is required"
    save_model_path = osp.join(flags.save_model_dir, model_save_suffix)
    pretrained_model_pathes = {
        "EMBEDDING": osp.join(flags.pretrain_model_dir, model_save_suffixes["pre_train_cl_model"]),
        "T_S": osp.join(flags.pretrain_model_dir, model_save_suffixes["pre_train_cl_model"]),
        "T_D": osp.join(flags.pretrain_model_dir, model_save_suffixes["pre_train_cl_model"]),
        "SEQ_G_LSTM_1": osp.join(flags.pretrain_model_dir, model_save_suffixes["train_generator"]),
        "SEQ_G_LSTM_2": osp.join(flags.pretrain_model_dir, model_save_suffixes["train_generator"]),
    }
    generator_model = AdversarialDDGModel(init_modules=AdversarialDDGModel.stepC_modules)
    generator_model.build(stepC=True, restorer_tag_notifier=["EMBEDDING", "T_S", "T_D", "SEQ_G_LSTM"])
    generator_model.fit(save_model_path=save_model_path, pretrain_model_pathes=pretrained_model_pathes)

def train_cl_model(model_save_suffix=model_save_suffixes["train_cl_model"]):
    assert flags.pretrain_model_dir, "pretrain_model_dir is required"
    save_model_path = osp.join(flags.save_model_dir, model_save_suffix)
    pretrained_model_pathes = {
        "EMBEDDING": osp.join(flags.pretrain_model_dir, model_save_suffixes["pre_train_cl_model"]),
        "T_S": osp.join(flags.pretrain_model_dir, model_save_suffixes["pre_train_cl_model"]),
        "T_D": osp.join(flags.pretrain_model_dir, model_save_suffixes["pre_train_cl_model"]),
        "SEQ_G_LSTM_1": osp.join(flags.pretrain_model_dir, model_save_suffixes["train_topic_generator"]),
        "SEQ_G_LSTM_2": osp.join(flags.pretrain_model_dir, model_save_suffixes["train_topic_generator"]),
    }
    generator_model = AdversarialDDGModel(init_modules=AdversarialDDGModel.stepD_modules)
    generator_model.build(stepD=True, restorer_tag_notifier=["EMBEDDING", "T_S", "T_D", "SEQ_G_LSTM"])
    generator_model.fit(save_model_path=save_model_path, pretrain_model_pathes=pretrained_model_pathes)

def eval_generator(eval_batch_size = flags["eval_batch_size"], eval_topic_count = flags["eval_topic_count"],
                   eval_seq_length = flags["eval_seq_length"]):
    eval_from_vals = ["generator", "topic_generator"]
    assert flags.eval_from in eval_from_vals, "eval_from must be one of %s" % eval_from_vals
    if flags.eval_from == "generator":
        model_save_suffix = model_save_suffixes["train_generator"]
    else:
        model_save_suffix = model_save_suffixes["train_topic_generator"]
    save_model_path = osp.join(flags.save_model_dir, model_save_suffix)
    generator_model = AdversarialDDGModel(init_modules=AdversarialDDGModel.eval_graph_modules)
    generator_model.build(eval_seq=True, batch_size=eval_batch_size, topic_count=eval_topic_count,
                          seq_length=eval_seq_length)
    generator_model.eval(save_model_path=save_model_path)

def eval_cl_model():
    eval_from_vals = ["pretrain_cl", "final_cl"]
    assert flags.eval_from in eval_from_vals, "eval_from must be one of %s" % eval_from_vals
    if flags.eval_from == "generator":
        model_save_suffix = model_save_suffixes["train_generator"]
    else:
        model_save_suffix = model_save_suffixes["pre_train_cl_model"]
    save_model_path = osp.join(flags.save_model_dir, model_save_suffix)
    generator_model = AdversarialDDGModel(init_modules=AdversarialDDGModel.eval_cl_modules)
    generator_model.build(eval_cl=True)
    generator_model.eval(save_model_path=save_model_path)

if __name__ == "__main__":
    vocab_freqs = WordCounter().load(
        osp.join(flags["lm_inputs"]["datapath"], "imdb_word_freqs.pickle")).most_common_freqs(
        flags["lm_sequence"]["vocab_size"])
    flags.add_variable(name="vocab_freqs", value=vocab_freqs)
    if flags.step == "train_lm_model":
        train_lm_model()
    elif flags.step == "pretrain_cl_model":
        pre_train_cl_model()
    elif flags.step == "train_ae_model":
        train_ae_model()
    elif flags.step == "train_generator":
        train_generator()
    elif flags.step == "train_topic_generator":
        train_topic_generator()
    elif flags.step == "train_cl_model":
        train_cl_model()
    elif flags.step == "eval_generator":
        eval_generator()
    elif flags.step == "eval_cl_model":
        # "--inputs_eval_count_examples" is required
        eval_cl_model()