import os

from nanollm.tokenizer.train_tokenizer import TokenizerTrainer


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    trainer = TokenizerTrainer(input_path, vocab_size, special_tokens)
    return trainer.train()
