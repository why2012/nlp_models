import sys
sys.path.insert(0, ".")
from adversarial_net.models import AdversarialClassificationModel
from adversarial_net import arguments as flags
from adversarial_net.preprocessing import WordCounter
from adversarial_net import osp
flags.add_argument(name="save_model_dir", argtype=str, default="E:/kaggle/avito/imdb_testset/adversarial_net/model/adv_cl_model/adv_cl_model.ckpt")
flags.add_argument(name="pretrained_model_path", argtype=str, default="E:/kaggle/avito/imdb_testset/adversarial_net/model/lm_model/lm_model.ckpt")

if __name__ == "__main__":
    vocab_freqs = WordCounter().load(
        osp.join(flags["lm_inputs"]["datapath"], "imdb_word_freqs.pickle")).most_common_freqs(
        flags["lm_sequence"]["vocab_size"])
    flags.add_variable(name="vocab_freqs", value=vocab_freqs)
    adv_cl_model = AdversarialClassificationModel()
    adv_cl_model.build()
    adv_cl_model.fit(save_model_path=flags["save_model_dir"], pretrained_model_path=flags["pretrained_model_path"])