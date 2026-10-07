from common import DATA_PATH

from nanollm.tokenizer.pretokenizer import GPTPreTokenizer
from nanollm.tokenizer.tokenizer import BytePairTokenizer


def test_merge():
    pretoknizer = GPTPreTokenizer(DATA_PATH / "test_corpus.txt")
    tokenizer = BytePairTokenizer(257 + 6, ["<|endoftext|>"])
    vocab, merges = tokenizer.train(pretoknizer)
    reference_merges = [
        (b"s", b"t"),
        (b"e", b"st"),
        (b"o", b"w"),
        (b"l", b"ow"),
        (b"w", b"est"),
        (b"n", b"e"),
    ]
    reference_vocab = {
        256: b"<|endoftext|>",
        257: b"st",
        258: b"est",
        259: b"ow",
        260: b"low",
        261: b"west",
        262: b"ne",
    }
    assert all(vocab[i] == reference_vocab[i] for i in range(256, 263))
    assert merges == reference_merges
