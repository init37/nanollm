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
    def __init__(self, vocab_size: int, special_tokens: list[str]) -> None:
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
        map_pretoken = {key: key for key in freq_table}
        pair_freq_table = {}
        for key in freq_table:
            for i in range(len(key) - 1):
                pair = (key[i], key[i + 1])
                if pair in pair_freq_table:
                    pair_freq_table[pair][key] = pair_freq_table[pair].get(key, 0) + 1
                else:
                    pair_freq_table[pair] = {key: 1}
        while len(self.vocab) < self.vocab_size:
            max_pair = None
            for pair, pretokens in pair_freq_table.items():
                value = sum(
                    freq_table[pretoken] * mult for pretoken, mult in pretokens.items()
                )
                if max_pair is None:
                    max_pair = (pair, value)
                else:
                    if max_pair[1] < value:
                        max_pair = (pair, value)
                    if max_pair[1] == value and max_pair[0] > pair:
                        max_pair = (pair, value)
            if max_pair is None:
                break
            self.vocab[len(self.vocab)] = (
                self.vocab[max_pair[0][0]] + self.vocab[max_pair[0][1]]
            )
            merges.append((self.vocab[max_pair[0][0]], self.vocab[max_pair[0][1]]))
            for p in list(pair_freq_table[max_pair[0]]):
                pretoken = map_pretoken[p]
                i = 0
                while i < len(pretoken) - 1:
                    if (
                        max_pair[0][0] == pretoken[i]
                        and max_pair[0][1] == pretoken[i + 1]
                    ):
                        if i > 0:
                            temp = (pretoken[i - 1], pretoken[i])
                            if temp in pair_freq_table and p in pair_freq_table[temp]:
                                pair_freq_table[temp][p] -= 1
                                if pair_freq_table[temp][p] == 0:
                                    del pair_freq_table[temp][p]
                        if i + 2 < len(pretoken):
                            temp = (pretoken[i + 1], pretoken[i + 2])
                            if temp in pair_freq_table and p in pair_freq_table[temp]:
                                pair_freq_table[temp][p] -= 1
                                if pair_freq_table[temp][p] == 0:
                                    del pair_freq_table[temp][p]
                        new_token = self.vocab[len(self.vocab) - 1]
                        merged = (
                            *pretoken[:i],
                            new_token,
                            *pretoken[i + 2 :],
                        )
                        map_pretoken[p] = merged
                        pretoken = merged
                        if i > 0:
                            temp = (pretoken[i - 1], pretoken[i])
                            if temp in pair_freq_table:
                                pair_freq_table[temp][p] = (
                                    pair_freq_table[temp].get(p, 0) + 1
                                )
                            else:
                                pair_freq_table[temp] = {p: 1}
                        if i + 1 < len(pretoken):
                            temp = (pretoken[i], pretoken[i + 1])
                            if temp in pair_freq_table:
                                pair_freq_table[temp][p] = (
                                    pair_freq_table[temp].get(p, 0) + 1
                                )
                            else:
                                pair_freq_table[temp] = {p: 1}
                    i += 1
                i = 0
            del pair_freq_table[pair]
        return merges
