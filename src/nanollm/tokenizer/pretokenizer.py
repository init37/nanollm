import os
from abc import ABC, abstractmethod
from typing import BinaryIO

import regex as re


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    if not isinstance(split_special_token, bytes):
        raise TypeError("Special token must be bytes type")
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))


class BytePreTokenizer(ABC):
    input_path: str | os.PathLike

    @abstractmethod
    def pre_tokenize(
        self, special_tokens: list[str]
    ) -> dict[tuple[bytes, ...], int]: ...


class GPTPreTokenizer(BytePreTokenizer):
    def __init__(self, input_path: str | os.PathLike) -> None:
        self.input_path = input_path
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def pre_tokenize(self, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
        freq_table = {}
        with open(self.input_path) as file:
            while line := file.readline():
                for special in special_tokens:
                    line = line.replace(special, "")
                for chunk in re.finditer(self.PAT, line):
                    pretoken = tuple(
                        bytes([byte]) for byte in chunk.group().encode("utf-8")
                    )
                    freq_table[pretoken] = freq_table.get(pretoken, 0) + 1
        return freq_table
