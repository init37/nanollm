import cProfile
import os
import pathlib

from nanollm.tokenizer.pretokenizer import GPTPreTokenizer
from nanollm.tokenizer.tokenizer import BytePairTokenizer

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "data"


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pretokenizer = GPTPreTokenizer(input_path)
    tokenizer = BytePairTokenizer(vocab_size, special_tokens)
    return tokenizer.train(pretokenizer)


input_path = DATA_PATH / "corpus.en"
profiler = cProfile.Profile()
profiler.enable()
run_train_bpe(
    input_path=input_path,
    vocab_size=500,
    special_tokens=["<|endoftext|>"],
)
profiler.disable()
profiler.dump_stats("train_bpe.prof")
