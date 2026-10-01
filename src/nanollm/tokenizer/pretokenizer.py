import os
from abc import ABC, abstractmethod

import regex as re


class BytePreTokenizer(ABC):
    input_path: str | os.PathLike

    @abstractmethod
    def pre_tokenize(self) -> dict[tuple[bytes, ...], int]: ...


class GPTPreTokenizer(BytePreTokenizer):
    def __init__(self, input_path: str | os.PathLike) -> None:
        self.input_path = input_path
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def pre_tokenize(self) -> dict[tuple[bytes, ...], int]:
        freq_table = {}
        with open(self.input_path) as file:
            while line := file.readline():
                for chunk in re.finditer(self.PAT, line):
                    pretoken = tuple(
                        bytes([byte]) for byte in chunk.group().encode("utf-8")
                    )
                    freq_table[pretoken] = freq_table.get(pretoken, 0) + 1
        return freq_table
