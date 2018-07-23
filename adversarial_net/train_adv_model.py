import sys
sys.path.insert(0, ".")
from adversarial_net.models import LanguageModel, AutoEncoderModel
from adversarial_net.AdversarialDDGModel import AdversarialDDGModel
from adversarial_net.VirtualAdversarialDDGModel import VirtualAdversarialDDGModel
from adversarial_net.AdversarialSummaryModel import AdversarialSummaryModel
from adversarial_net.SummaryModel import SummaryModel
from adversarial_net import arguments as flags
from adversarial_net.preprocessing import WordCounter
from adversarial_net import osp
from adversarial_net.utils import getLogger
logger = getLogger("train_model")
training_step_vals = ["train_lm_model", "pretrain_cl_model", "train_ae_model", "train_generator", "train_topic_generator",
                      "train_cl_model", "eval_generator", "eval_cl_model", "eval_lm_model", "eval_ae_model",
                      "train_summary_model", "eval_summary_model", "train_summary_cl_model", "eval_summary_cl_model"]
model_save_suffixes = {
    "train_lm_model": "lm_model/lm_model.ckpt",
    "pre_train_cl_model": "adv_cl_model/adv_cl_model.ckpt",
    "train_ae_model": "ae_model/ae_model.ckpt",
    "train_generator": "generator/generator.ckpt",
    "train_topic_generator": "topic_generator/topic_generator.ckpt",
    "train_cl_model": "final_cl_model/final_cl_model.ckpt",
    "train_summary_model": "summary_model/summary_model.ckpt",
    "train_summary_cl_model": "summary_cl_model/summary_cl_model.ckpt",
}
class ModelPrefixManager(object):
    NO_PREIFX_TAG = "[no_prefix]"
    def __init__(self, suffix_map):
        self.suffix_map = suffix_map
        self.suffix_choice = None
    def __getitem__(self, item):
        if flags["model_prefix"] is None and flags["adv_type"] != "adv":
            logger.error("model_prefix is required when adv_type != adv (default)")
            exit(0)
        no_prefix_tag = False
        if item.startswith(self.NO_PREIFX_TAG):
            item = item[len(self.NO_PREIFX_TAG):]
            no_prefix_tag = True
        if flags["model_prefix"] is None:
            if self.suffix_choice is None:
                self.suffix_choice = input("Model prefix is not provided, continue (Y/N): ").strip().upper()
            if self.suffix_choice == "Y":
                return self.suffix_map[item]
            else:
                exit(0)
        else:
            if no_prefix_tag:
                return self.suffix_map[item]
            else:
                return "{prefix}_{suffix}".format(prefix=flags["model_prefix"], suffix=self.suffix_map[item])
model_save_suffixes = ModelPrefixManager(suffix_map=model_save_suffixes)
def training_step(value):
    assert value in training_step_vals, "step is one of %s" % training_step_vals
    return value
def eval_from(value):
    eval_from_vals = ["generator", "topic_generator", "pretrain_cl", "final_cl"]
    assert value in eval_from_vals, "step is one of %s" % eval_from_vals
    return value
def adv_type(value):
    adv_types = ["adv", "vir_adv"]
    assert value in adv_types, "adv_type is one of %s" % adv_types
    return value
flags.add_argument(name="step", argtype=training_step)
flags.add_argument(name="save_model_dir", argtype=str)
flags.add_argument(name="pretrain_model_dir", argtype=str, default=None)
flags.add_argument(name="eval_from", argtype=eval_from, default="generator")
flags.add_argument(name="eval_batch_size", argtype=int, default=2)
flags.add_argument(name="eval_topic_count", argtype=int, default=2)
flags.add_argument(name="eval_seq_length", argtype=int, default=200)
# lm/ae model args
flags.add_argument(name="no_loss_sampler", argtype=bool, default=False)
flags.add_argument(name="hard_mode", argtype=bool, default=False)
flags.add_argument(name="forget_bias", argtype=float, default=0.0)
# model prefix
flags.add_argument(name="model_prefix", argtype=str, default=None)
# adversarial training type
flags.add_argument(name="adv_type", argtype=adv_type, default="adv")

flags.add_argument(name="inputs_docs_path", argtype=str, default="E:/kaggle/avito/imdb_testset/adversarial_net/data/summary/train/train.article.txt")
flags.add_argument(name="inputs_docs_batch_size", argtype=int, default=5)

# training process         (->embed)
#                  |--> training lm_model |         (->embed)                 (lock embed)              (lock embed)               (->embed)
# start training --|       (lock embed)   |--> pre-training cl_model --> | training generator --> training topic generator --> | re-train cl_model
#                  |--> training ae_model |                              -> eval generator                                     -> eval cl model

def train_lm_model(model_save_suffix = model_save_suffixes["train_lm_model"]):
    save_model_path = osp.join(flags.save_model_dir, model_save_suffix)
    lm_model = LanguageModel()
    lm_model.build(use_sampler=not flags["no_loss_sampler"], hard_mode=flags["hard_mode"], forget_bias=flags["forget_bias"])
    lm_model.fit(save_model_path=save_model_path)

def pre_train_cl_model(model_save_suffix = model_save_suffixes["pre_train_cl_model"]):
    assert flags.pretrain_model_dir, "pretrain_model_dir is required"
    save_model_path = osp.join(flags.save_model_dir, model_save_suffix)
    pretrained_model_pathes = {
        "EMBEDDING": osp.join(flags.pretrain_model_dir, model_save_suffixes["[no_prefix]train_lm_model"]),
        "T_S": osp.join(flags.pretrain_model_dir, model_save_suffixes["[no_prefix]train_lm_model"])
    }
    if flags["adv_type"] == "adv":
        adv_cl_model = AdversarialDDGModel(init_modules=AdversarialDDGModel.stepB_modules)
    elif flags["adv_type"] == "vir_adv":
        adv_cl_model = VirtualAdversarialDDGModel(init_modules=VirtualAdversarialDDGModel.stepB_modules)
    else:
        raise Exception("Unknow adv_type: %s" % flags["adv_type"])
    adv_cl_model.build(stepB=True, restorer_tag_notifier=[])
    adv_cl_model.fit(save_model_path=save_model_path, pretrain_model_pathes=pretrained_model_pathes)

def train_ae_model(model_save_suffix=model_save_suffixes["train_ae_model"]):
    assert flags.pretrain_model_dir, "pretrain_model_dir is required"
    save_model_path = osp.join(flags.save_model_dir, model_save_suffix)
    pretrained_model_path = osp.join(flags.pretrain_model_dir, model_save_suffixes["train_lm_model"])
    ae_model = AutoEncoderModel(lock_embedding=True)
    ae_model.build(use_sampler=not flags["no_loss_sampler"], hard_mode=flags["hard_mode"], forget_bias=flags["forget_bias"])
    ae_model.fit(save_model_path=save_model_path, pretrained_model_path=pretrained_model_path)

def train_generator(model_save_suffix=model_save_suffixes["[no_prefix]train_generator"]):
    assert flags.pretrain_model_dir, "pretrain_model_dir is required"
    save_model_path = osp.join(flags.save_model_dir, model_save_suffix)
    pretrained_model_pathes = {
        "EMBEDDING": osp.join(flags.pretrain_model_dir, model_save_suffixes["[no_prefix]pre_train_cl_model"]),
        "FG_S": osp.join(flags.pretrain_model_dir, model_save_suffixes["[no_prefix]train_lm_model"]),
        "SEQ_G_LSTM_1": osp.join(flags.pretrain_model_dir, model_save_suffixes["train_lm_model"]),
        "SEQ_G_LSTM_2": osp.join(flags.pretrain_model_dir, model_save_suffixes["train_ae_model"]),
        "RNN_TO_EMBEDDING": osp.join(flags.pretrain_model_dir, model_save_suffixes["train_lm_model"]),
    }
    if flags["adv_type"] == "adv":
        generator_model = AdversarialDDGModel(init_modules=AdversarialDDGModel.stepA_modules)
    elif flags["adv_type"] == "vir_adv":
        generator_model = VirtualAdversarialDDGModel(init_modules=VirtualAdversarialDDGModel.stepA_modules)
    else:
        raise Exception("Unknow adv_type: %s" % flags["adv_type"])
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
        "RNN_TO_EMBEDDING": osp.join(flags.pretrain_model_dir, model_save_suffixes["train_generator"]),
    }
    if flags["adv_type"] == "adv":
        generator_model = AdversarialDDGModel(init_modules=AdversarialDDGModel.stepC_modules)
    elif flags["adv_type"] == "vir_adv":
        generator_model = VirtualAdversarialDDGModel(init_modules=VirtualAdversarialDDGModel.stepC_modules)
    else:
        raise Exception("Unknow adv_type: %s" % flags["adv_type"])
    generator_model.build(stepC=True, restorer_tag_notifier=["EMBEDDING", "T_S", "T_D", "SEQ_G_LSTM", "RNN_TO_EMBEDDING"])
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
        "RNN_TO_EMBEDDING": osp.join(flags.pretrain_model_dir, model_save_suffixes["train_topic_generator"]),
    }
    if flags["adv_type"] == "adv":
        generator_model = AdversarialDDGModel(init_modules=AdversarialDDGModel.stepD_modules)
    elif flags["adv_type"] == "vir_adv":
        generator_model = VirtualAdversarialDDGModel(init_modules=VirtualAdversarialDDGModel.stepD_modules)
    else:
        raise Exception("Unknow adv_type: %s" % flags["adv_type"])
    generator_model.build(stepD=True, restorer_tag_notifier=["EMBEDDING", "T_S", "T_D", "SEQ_G_LSTM", "RNN_TO_EMBEDDING"])
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
    if flags.eval_from == "final_cl":
        model_save_suffix = model_save_suffixes["train_cl_model"]
    else:
        model_save_suffix = model_save_suffixes["pre_train_cl_model"]
    save_model_path = osp.join(flags.save_model_dir, model_save_suffix)
    generator_model = AdversarialDDGModel(init_modules=AdversarialDDGModel.eval_cl_modules)
    generator_model.build(eval_cl=True)
    generator_model.eval(save_model_path=save_model_path)

def eval_lm_model(model_save_suffix = model_save_suffixes["train_lm_model"]):
    save_model_path = osp.join(flags.save_model_dir, model_save_suffix)
    lm_model = LanguageModel()
    lm_model.eval(save_model_path=save_model_path)

def eval_ae_model(model_save_suffix = model_save_suffixes["train_ae_model"]):
    save_model_path = osp.join(flags.save_model_dir, model_save_suffix)
    ae_model = AutoEncoderModel()
    ae_model.eval(save_model_path=save_model_path)

# lm_model -> summary_model -> cl_summary_model

def train_summary_model(model_save_suffix = model_save_suffixes["train_summary_model"]):
    save_model_path = osp.join(flags.save_model_dir, model_save_suffix)
    pretrained_model_path = None
    if flags.pretrain_model_dir:
        pretrained_model_path = osp.join(flags.pretrain_model_dir, model_save_suffixes["train_lm_model"])
    summary_model = SummaryModel()
    summary_model.build()
    summary_model.fit(save_model_path=save_model_path, pretrained_model_path=pretrained_model_path)

def eval_summary_model(model_save_suffix = model_save_suffixes["train_summary_model"]):
    save_model_path = osp.join(flags.save_model_dir, model_save_suffix)
    read_docs = []
    inputs_docs = []
    with open(flags.inputs_docs_path, "r", encoding="utf-8") as f:
        for i in range(1000):
            read_docs.append(f.readline())
    import random
    for i in range(flags.inputs_docs_batch_size):
        inputs_docs.append(random.choice(read_docs))
    summary_model = SummaryModel()
    summary_model.eval(inputs_docs=inputs_docs, save_model_path=save_model_path)

def train_summary_cl_model(model_save_suffix = model_save_suffixes["train_summary_cl_model"]):
    assert flags.pretrain_model_dir, "pretrain_model_dir is required"
    save_model_path = osp.join(flags.save_model_dir, model_save_suffix)
    pretrained_model_pathes = {
        "EMBEDDING": osp.join(flags.pretrain_model_dir, model_save_suffixes["train_summary_model"]),
        "T_S": osp.join(flags.pretrain_model_dir, model_save_suffixes["train_lm_model"]),
        "SUMMARY": osp.join(flags.pretrain_model_dir, model_save_suffixes["train_summary_model"]),
    }
    if flags["adv_type"] == "adv":
        adv_cl_model = AdversarialSummaryModel()
    elif flags["adv_type"] == "vir_adv":
        raise Exception("Unimplement")
    else:
        raise Exception("Unknow adv_type: %s" % flags["adv_type"])
    adv_cl_model.build(training=True, restorer_tag_notifier=["EMBEDDING", "SUMMARY"])
    adv_cl_model.fit(save_model_path=save_model_path, pretrain_model_pathes=pretrained_model_pathes)

def eval_summary_cl_model():
    model_save_suffix = model_save_suffixes["train_summary_cl_model"]
    save_model_path = osp.join(flags.save_model_dir, model_save_suffix)
    generator_model = AdversarialSummaryModel()
    generator_model.build(eval_cl=True)
    generator_model.eval(save_model_path=save_model_path)

# intersection count between classi word_freqs and summary word_freqs: {10000: 9652, 20000: 18673, 30000: 26590, 40000: 33259, 50000: 38737, 60000: 43262, 70000: 46964, 80000: 49788, 86934: 51515}
if __name__ == "__main__":
    if flags.step == "train_summary_model" or flags.step == "eval_summary_model":
        inersect_count = []
        vocab_freqs = WordCounter().load_and_merge(
            osp.join(flags["lm_inputs"]["datapath"], "%s_word_freqs.pickle" % flags["lm_inputs"]["dataset"]),
            osp.join(flags["lm_inputs"]["datapath"], "summary_word_freqs.pickle"),
            max_words=list(range(0, flags["inputs"]["vocab_size"], 10000))[1:] + [flags["inputs"]["vocab_size"]],
            return_cache=inersect_count
        ).most_common_freqs(flags["lm_sequence"]["vocab_size"])
        inersect_count = inersect_count[0]
        logger.info("intersection count between classi word_freqs and summary word_freqs: %s" % inersect_count)
    else:
        vocab_freqs = WordCounter().load(
            osp.join(flags["lm_inputs"]["datapath"], "%s_word_freqs.pickle" % flags["lm_inputs"]["dataset"])).most_common_freqs(
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
    elif flags.step == "eval_lm_model":
        eval_lm_model()
    elif flags.step == "eval_ae_model":
        eval_ae_model()
    elif flags.step == "train_summary_model":
        train_summary_model()
    elif flags.step == "eval_summary_model":
        eval_summary_model()
    elif flags.step == "train_summary_cl_model":
        train_summary_cl_model()
    elif flags.step == "eval_summary_cl_model":
        eval_summary_cl_model()