from common import DATA_PATH

from nanollm.tokenizer.tokenizer import BytePairTokenizer
from nanollm.tokenizer.train_tokenizer import TokenizerTrainer


def test_merge():
    trainer = TokenizerTrainer(
        DATA_PATH / "test_corpus.txt", 257 + 6, ["<|endoftext|>"]
    )
    freq_table = trainer.pretokenize()
    tokenizer = BytePairTokenizer(257 + 6, ["<|endoftext|>"])
    vocab, merges = tokenizer.train(freq_table)
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
    for m, r in zip(merges, reference_merges):
        if m != r:
            print(m, r)
    assert all(vocab[i] == reference_vocab[i] for i in range(256, 263))
    assert merges == reference_merges


def test_double_char_edge_case():
    trainer = TokenizerTrainer(
        DATA_PATH / "double_char_edge_case.txt", 257 + 1, ["<|endoftext|>"]
    )
    freq_table = trainer.pretokenize()
    tokenizer = BytePairTokenizer(257 + 1, ["<|endoftext|>"])
    vocab, merges = tokenizer.train(freq_table)
    reference_merges = [
        (b"a", b"a"),
    ]
    reference_vocab = {256: b"<|endoftext|>", 257: b"aa"}
    assert all(vocab[i] == reference_vocab[i] for i in range(256, 258))
    assert merges == reference_merges


def test_double_char_edge_case_two():
    trainer = TokenizerTrainer(
        DATA_PATH / "double_char_edge_case.txt", 257 + 2, ["<|endoftext|>"]
    )
    freq_table = trainer.pretokenize()
    tokenizer = BytePairTokenizer(257 + 2, ["<|endoftext|>"])
    vocab, merges = tokenizer.train(freq_table)
    reference_merges = [
        (b"a", b"a"),
        (b"aa", b"aa"),
    ]
    reference_vocab = {256: b"<|endoftext|>", 257: b"aa", 258: b"aaaa"}
    assert all(vocab[i] == reference_vocab[i] for i in range(256, 259))
    assert merges == reference_merges
