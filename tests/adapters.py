import os

from nanollm.tokenizer.tokenizer import BytePairTokenizer
from nanollm.tokenizer.train_tokenizer import TokenizerTrainer


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    trainer = TokenizerTrainer(input_path, vocab_size, special_tokens)
    freq_table = trainer.pretokenize()
    tokenizer = BytePairTokenizer(vocab_size, special_tokens)
    return tokenizer.train(freq_table)
