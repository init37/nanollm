import os
from itertools import pairwise
from multiprocessing import Pool

from nanollm.tokenizer.pretokenizer import (
    GPTPreTokenizer,
    find_chunk_boundaries,
)


class TokenizerTrainer:
    def __init__(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        num_processes: int = 4,
    ) -> None:
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.num_processes = num_processes

    def pretokenize(
        self,
        boundaries_token: bytes = b"<|endoftext|>",
    ) -> dict[tuple[bytes, ...], int]:
        final_freq_table: dict[tuple[bytes, ...], int] = {}
        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, boundaries_token)
        chunks = list(pairwise(boundaries))
        tasks = [(bound, self.input_path, self.special_tokens) for bound in chunks]
        with Pool(self.num_processes) as pool:
            freq_tables = pool.starmap(
                self.run_pretokenizer,
                tasks,
            )
        for freq_table in freq_tables:
            for key, value in freq_table.items():
                final_freq_table[key] = final_freq_table.get(key, 0) + value
        return final_freq_table

    @staticmethod
    def run_pretokenizer(
        bound: tuple[int, int], input_path: str | os.PathLike, special_tokens: list[str]
    ):
        start, end = bound
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            pretokenizer = GPTPreTokenizer(chunk, special_tokens)
        return pretokenizer.pre_tokenize()
