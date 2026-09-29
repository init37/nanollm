from abc import ABC, abstractmethod

from nanollm.tokenizer.pretokenizer import BytePreTokenizer


class ByteTokenizer(ABC):
    input_path: str
    vocab_size: int
    special_tokens: list[str]

    @abstractmethod
    def train(
        self, pretokenizer: BytePreTokenizer
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: ...


class BytePairTokenizer(ByteTokenizer):
    def __init__(
        self, input_path: str, vocab_size: int, special_tokens: list[str]
    ) -> None:
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, special_token in enumerate(special_tokens):
            self.vocab[256 + i] = special_token.encode("utf-8")

    def train(
        self, pretokenizer: BytePreTokenizer
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        freq_table = pretokenizer.pre_tokenize()
        merges = self.merge(freq_table)
        return self.vocab, merges

    def merge(
        self, freq_table: dict[tuple[bytes, ...], int]
    ) -> list[tuple[bytes, bytes]]:
        merges = []
        while len(self.vocab) < self.vocab_size:
            pair_freq_table = {}
            for key, value in freq_table.items():
                for i in range(len(key) - 1):
                    pair = (key[i], key[i + 1])
                    pair_freq_table[pair] = pair_freq_table.get(pair, 0) + value
            print(pair_freq_table)

        return merges
