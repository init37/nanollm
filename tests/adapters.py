import os

from nanollm.tokenizer.tokenizer import Tokenizer
from nanollm.tokenizer.train_tokenizer import TokenizerTrainer


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[bytes],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    trainer = TokenizerTrainer(input_path, vocab_size, special_tokens)
    return trainer.train()


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    special_tokens_bytes = None
    if special_tokens:
        special_tokens_bytes = [st.encode("UTF-8") for st in special_tokens]
    return Tokenizer(vocab, merges, special_tokens_bytes)
