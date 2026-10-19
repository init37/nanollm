import cProfile
import os
import pathlib

from nanollm.tokenizer.tokenizer import BytePairTokenizer
from nanollm.tokenizer.train_tokenizer import TokenizerTrainer

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "data"


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    trainer = TokenizerTrainer(input_path, vocab_size, special_tokens, num_processes=16)
    freq_table = trainer.pretokenize()
    tokenizer = BytePairTokenizer(vocab_size, special_tokens)
    return tokenizer.train(freq_table)


def main():
    input_path = DATA_PATH / "big" / "TinyStoriesV2-GPT4-train.txt"
    profiler = cProfile.Profile()
    profiler.enable()
    run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    profiler.disable()
    profiler.dump_stats("train_bpe.prof")


if __name__ == "__main__":
    main()
