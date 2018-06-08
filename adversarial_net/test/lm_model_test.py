import sys
sys.path.insert(0, ".")
from adversarial_net.models import LanguageModel
from adversarial_net import arguments as flags
from adversarial_net.preprocessing import WordCounter
from adversarial_net import osp
flags.add_argument(name="save_model_dir", argtype=str, default="E:/kaggle/avito/imdb_testset/adversarial_net/model/lm_model/lm_model.ckpt")

if __name__ == "__main__":
    vocab_freqs = WordCounter().load(
        osp.join(flags["lm_inputs"]["datapath"], "imdb_word_freqs.pickle")).most_common_freqs(
        flags["lm_sequence"]["vocab_size"])
    flags.add_variable(name="vocab_freqs", value=vocab_freqs)
    lm_model = LanguageModel()
    lm_model.build()
    lm_model.fit(save_model_path=flags["save_model_dir"])