import os

from nanollm.tokenizer.pretokenizer import GPTPreTokenizer
from nanollm.tokenizer.tokenizer import BytePairTokenizer


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pretokenizer = GPTPreTokenizer(input_path)
    tokenizer = BytePairTokenizer(vocab_size, special_tokens)
    return tokenizer.train(pretokenizer)
