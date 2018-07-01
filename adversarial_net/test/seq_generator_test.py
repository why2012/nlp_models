import sys
sys.path.insert(0, ".")
from adversarial_net.AdversarialDDGModel import AdversarialDDGModel
from adversarial_net.preprocessing import WordCounter
from adversarial_net import osp
from adversarial_net import arguments as flags

if __name__ == "__main__":
    vocab_freqs = WordCounter().load(
        osp.join(flags["inputs"]["datapath"], "imdb_word_freqs.pickle")).most_common_freqs(
        flags["lm_sequence"]["vocab_size"])
    flags.add_variable(name="vocab_freqs", value=vocab_freqs)
    adv_model = AdversarialDDGModel(init_modules=AdversarialDDGModel.eval_graph_modules)
    adv_model.build(eval_seq=True, batch_size=2, topic_count=2, seq_length=200)
    adv_model.eval(None)
