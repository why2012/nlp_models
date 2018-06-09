import sys
sys.path.insert(0, ".")
from adversarial_net.models import AutoEncoderModel
from adversarial_net import arguments as flags
from adversarial_net.preprocessing import WordCounter
from adversarial_net import osp
flags.add_argument(name="save_model_dir", argtype=str, default="E:/kaggle/avito/imdb_testset/adversarial_net/model/ae_model/ae_model.ckpt")

if __name__ == "__main__":
    vocab_freqs = WordCounter().load(
        osp.join(flags["ae_inputs"]["datapath"], "imdb_word_freqs.pickle")).most_common_freqs(
        flags["ae_sequence"]["vocab_size"])
    flags.add_variable(name="vocab_freqs", value=vocab_freqs)
    ae_model = AutoEncoderModel()
    ae_model.build()
    ae_model.fit(save_model_path=flags["save_model_dir"])