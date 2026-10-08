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
        self.special_tokens = special_tokens
        for i, special_token in enumerate(special_tokens):
            self.vocab[256 + i] = special_token.encode("utf-8")

    def train(
        self, pretokenizer: BytePreTokenizer
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        freq_table = pretokenizer.pre_tokenize(self.special_tokens)
        merges = self.merge(freq_table)
        return self.vocab, merges

    def merge(
        self, freq_table: dict[tuple[bytes, ...], int]
    ) -> list[tuple[bytes, bytes]]:
        merges: list[tuple[bytes, bytes]] = []
        map_pretoken = {key: key for key in freq_table}
        pair_freq_table: dict[tuple[bytes, bytes], dict[tuple[bytes, ...], int]] = {}
        pair_freq: dict[tuple[bytes, bytes], int] = {}
        for key in freq_table:
            for i in range(len(key) - 1):
                pair = (key[i], key[i + 1])
                if pair in pair_freq_table:
                    pair_freq_table[pair][key] = pair_freq_table[pair].get(key, 0) + 1
                else:
                    pair_freq_table[pair] = {key: 1}
        for pair, pretokens in pair_freq_table.items():
            value = sum(
                freq_table[pretoken] * mult for pretoken, mult in pretokens.items()
            )
            pair_freq[pair] = value

        while len(self.vocab) < self.vocab_size:
            max_pair: tuple[tuple[bytes, bytes], int] | None = None
            if pair_freq:
                max_key = max(pair_freq, key=lambda k: (pair_freq[k], k))
                max_pair = (max_key, pair_freq[max_key])
            else:
                break
            self.vocab[len(self.vocab)] = max_pair[0][0] + max_pair[0][1]
            merges.append((max_pair[0][0], max_pair[0][1]))
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
                                pair_freq[temp] -= freq_table[p]
                                if pair_freq_table[temp][p] == 0:
                                    del pair_freq_table[temp][p]
                                    if not pair_freq_table[temp]:
                                        del pair_freq_table[temp]
                                        del pair_freq[temp]
                        if i + 2 < len(pretoken):
                            temp = (pretoken[i + 1], pretoken[i + 2])
                            if temp in pair_freq_table and p in pair_freq_table[temp]:
                                pair_freq_table[temp][p] -= 1
                                pair_freq[temp] -= freq_table[p]
                                if pair_freq_table[temp][p] == 0:
                                    del pair_freq_table[temp][p]
                                    if not pair_freq_table[temp]:
                                        del pair_freq_table[temp]
                                        del pair_freq[temp]
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
                                pair_freq[temp] += freq_table[p]
                            else:
                                pair_freq_table[temp] = {p: 1}
                                pair_freq[temp] = freq_table[p]
                        if i + 1 < len(pretoken):
                            temp = (pretoken[i], pretoken[i + 1])
                            if temp in pair_freq_table:
                                pair_freq_table[temp][p] = (
                                    pair_freq_table[temp].get(p, 0) + 1
                                )
                                pair_freq[temp] += freq_table[p]
                            else:
                                pair_freq_table[temp] = {p: 1}
                                pair_freq[temp] = freq_table[p]
                    i += 1
                i = 0
            if max_pair[0] in pair_freq_table:
                del pair_freq_table[max_pair[0]]
                del pair_freq[max_pair[0]]
        return merges
