import os
from itertools import pairwise
from multiprocessing import Pool

from nanollm.tokenizer.pretokenizer import (
    GPTPreTokenizer,
    find_chunk_boundaries,
)
from nanollm.tokenizer.tokenizer import BytePairTokenizerTrainer


class TokenizerTrainer:
    def __init__(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[bytes],
        num_workers: int = 4,
        chunk_size_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        if num_workers < 1:
            raise ValueError("num_processes must be at least 1")
        if chunk_size_bytes < 1:
            raise ValueError("chunk_size_bytes must be at least 1")
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.num_workers = num_workers
        self.chunk_size_bytes = chunk_size_bytes

    def pretokenize(
        self,
        boundaries_token: bytes = b"<|endoftext|>",
    ) -> dict[tuple[bytes, ...], int]:
        final_freq_table: dict[tuple[bytes, ...], int] = {}
        with open(self.input_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            desired_num_chunks = max(
                self.num_workers,
                (file_size + self.chunk_size_bytes - 1) // self.chunk_size_bytes,
            )
            boundaries = find_chunk_boundaries(f, desired_num_chunks, boundaries_token)
        chunks = list(pairwise(boundaries))
        tasks = [(bound, self.input_path, self.special_tokens) for bound in chunks]
        with Pool(self.num_workers) as pool:
            for freq_table in pool.imap_unordered(_run_pretokenizer_task, tasks):
                for key, value in freq_table.items():
                    final_freq_table[key] = final_freq_table.get(key, 0) + value
                del freq_table
        return final_freq_table

    @staticmethod
    def run_pretokenizer(
        bound: tuple[int, int],
        input_path: str | os.PathLike,
        special_tokens: list[bytes],
    ):
        start, end = bound
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            pretokenizer = GPTPreTokenizer(chunk, special_tokens)
        return pretokenizer.pre_tokenize_train()

    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        freq_table = self.pretokenize()
        tokenizer = BytePairTokenizerTrainer(self.vocab_size, self.special_tokens)
        return tokenizer.train(freq_table)


def _run_pretokenizer_task(
    task: tuple[tuple[int, int], str | os.PathLike, list[bytes]],
) -> dict[tuple[bytes, ...], int]:
    return TokenizerTrainer.run_pretokenizer(*task)


if __name__ == "__main__":
    import pathlib
    import pickle

    DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "data"
    db = "owt_train.txt"
    input_path = DATA_PATH / "big" / db
    cache_path = DATA_PATH / "processed"

    vocab_size = 32000
    special_tokens = [b"<|endoftext|>"]

    trainer = TokenizerTrainer(
        input_path,
        vocab_size,
        special_tokens,
        num_workers=4,
        chunk_size_bytes=32 * 1024 * 1024,
    )
    freq_table = trainer.pretokenize()
    tokenizer = BytePairTokenizerTrainer(vocab_size, special_tokens)
    vocab, merges = tokenizer.train(freq_table)

    vocab_cache = cache_path / (db.split(".")[0] + "_vocab" + ".pkl")
    merges_cache = cache_path / (db.split(".")[0] + "_merges" + ".pkl")
    with vocab_cache.open("wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    with merges_cache.open("wb") as f:
        pickle.dump(merges, f, protocol=pickle.HIGHEST_PROTOCOL)
