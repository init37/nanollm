from __future__ import annotations

import json
import os
import resource
import sys
from collections.abc import Iterable, Iterator

import psutil
import pytest
import tiktoken
from adapters import get_tokenizer
from common import DATA_PATH, gpt2_bytes_to_unicode

from nanollm.tokenizer.pretokenizer import GPTPreTokenizer
from nanollm.tokenizer.tokenizer import BytePairTokenizerTrainer, Tokenizer
from nanollm.tokenizer.train_tokenizer import TokenizerTrainer

VOCAB_PATH = DATA_PATH / "reference" / "gpt2_vocab.json"
MERGES_PATH = DATA_PATH / "reference" / "gpt2_merges.txt"


class _ConcreteTokenizer(Tokenizer):
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8")


def test_encode_flattens_pretokenized_tokens():
    vocab = {i: bytes([i]) for i in range(256)}
    tokenizer = _ConcreteTokenizer(vocab, [], [])

    assert tokenizer.encode("hello world") == list(b"hello world")


def test_pretokenize_preserves_special_tokens():
    special_token = b"<|endoftext|>"
    pretokenizer = GPTPreTokenizer(
        f"hello{special_token.decode('utf-8')}world", [special_token]
    )

    assert pretokenizer.pre_tokenize() == [
        tuple(bytes([byte]) for byte in b"hello"),
        (special_token,),
        tuple(bytes([byte]) for byte in b"world"),
    ]


def test_merge():
    trainer = TokenizerTrainer(
        DATA_PATH / "test_corpus.txt", 257 + 6, [b"<|endoftext|>"]
    )
    freq_table = trainer.pretokenize()
    tokenizer = BytePairTokenizerTrainer(257 + 6, [b"<|endoftext|>"])
    vocab, merges = tokenizer.train(freq_table)
    reference_merges = [
        (b"s", b"t"),
        (b"e", b"st"),
        (b"o", b"w"),
        (b"l", b"ow"),
        (b"w", b"est"),
        (b"n", b"e"),
    ]
    reference_vocab = {
        256: b"<|endoftext|>",
        257: b"st",
        258: b"est",
        259: b"ow",
        260: b"low",
        261: b"west",
        262: b"ne",
    }
    for m, r in zip(merges, reference_merges):
        if m != r:
            print(m, r)
    assert all(vocab[i] == reference_vocab[i] for i in range(256, 263))
    assert merges == reference_merges


def test_double_char_edge_case():
    trainer = TokenizerTrainer(
        DATA_PATH / "double_char_edge_case.txt", 257 + 1, [b"<|endoftext|>"]
    )
    freq_table = trainer.pretokenize()
    tokenizer = BytePairTokenizerTrainer(257 + 1, [b"<|endoftext|>"])
    vocab, merges = tokenizer.train(freq_table)
    reference_merges = [
        (b"a", b"a"),
    ]
    reference_vocab = {256: b"<|endoftext|>", 257: b"aa"}
    assert all(vocab[i] == reference_vocab[i] for i in range(256, 258))
    assert merges == reference_merges


def test_double_char_edge_case_two():
    trainer = TokenizerTrainer(
        DATA_PATH / "double_char_edge_case.txt", 257 + 2, [b"<|endoftext|>"]
    )
    freq_table = trainer.pretokenize()
    tokenizer = BytePairTokenizerTrainer(257 + 2, [b"<|endoftext|>"])
    vocab, merges = tokenizer.train(freq_table)
    reference_merges = [
        (b"a", b"a"),
        (b"aa", b"aa"),
    ]
    reference_vocab = {256: b"<|endoftext|>", 257: b"aa", 258: b"aaaa"}
    assert all(vocab[i] == reference_vocab[i] for i in range(256, 259))
    assert merges == reference_merges


def test_pretokenize_chunk_sizes_match(tmp_path):
    input_path = tmp_path / "repeated_documents.txt"
    input_path.write_text(
        "alpha beta gamma <|endoftext|>\n" * 20,
        encoding="utf-8",
    )

    small_chunks = TokenizerTrainer(
        input_path,
        257,
        [b"<|endoftext|>"],
        num_workers=2,
        chunk_size_bytes=64,
    ).pretokenize()
    one_chunk = TokenizerTrainer(
        input_path,
        257,
        [b"<|endoftext|>"],
        num_workers=1,
        chunk_size_bytes=1024 * 1024,
    ).pretokenize()

    assert small_chunks == one_chunk


def memory_limit(max_mem):
    def decorator(f):
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            prev_limits = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(
                resource.RLIMIT_AS, (process.memory_info().rss + max_mem, -1)
            )
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                # Even if the function above fails (e.g., it exceeds the
                # memory limit), reset the memory limit back to the
                # previous limit so other tests aren't affected.
                resource.setrlimit(resource.RLIMIT_AS, prev_limits)

        return wrapper

    return decorator


def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return get_tokenizer(vocab, merges, special_tokens)


def test_roundtrip_empty():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = ""
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_empty_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = ""

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    tokenized_string = [tokenizer.decode([x]) for x in ids]
    assert tokenized_string == []

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_roundtrip_single_character():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "s"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_single_character_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "s"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    tokenized_string = [tokenizer.decode([x]) for x in ids]
    assert tokenized_string == ["s"]

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_roundtrip_single_unicode_character():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "🙃"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_single_unicode_character_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "🙃"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_roundtrip_ascii_string():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "Hello, how are you?"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_ascii_string_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Hello, how are you?"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    # assert ids == reference_ids

    tokenized_string = [tokenizer.decode([x]) for x in ids]
    assert tokenized_string == ["Hello", ",", " how", " are", " you", "?"]

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_roundtrip_unicode_string():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "Héllò hôw are ü? 🙃"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_unicode_string_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw are ü? 🙃"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_roundtrip_unicode_string_with_special_tokens():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    encoded_ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    # Ensure the special <|endoftext|> token is preserved
    assert tokenized_string.count("<|endoftext|>") == 3

    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_unicode_string_with_special_tokens_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"

    reference_ids = reference_tokenizer.encode(
        test_string, allowed_special={"<|endoftext|>"}
    )
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_overlapping_special_tokens():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    )
    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"

    ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in ids]
    # Ensure the double <|endoftext|><|endoftext|> is preserved as a single token
    assert tokenized_string.count("<|endoftext|>") == 1
    assert tokenized_string.count("<|endoftext|><|endoftext|>") == 1
    # Test roundtrip
    assert tokenizer.decode(ids) == test_string


def test_address_roundtrip():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(DATA_PATH / "address.txt") as f:
        corpus_contents = f.read()

    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


def test_address_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    corpus_path = DATA_PATH / "address.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents)
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


def test_german_roundtrip():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(DATA_PATH / "german.txt") as f:
        corpus_contents = f.read()

    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


def test_german_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    corpus_path = DATA_PATH / "german.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents)
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


def test_tinystories_sample_roundtrip():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(DATA_PATH / "tinystories_sample.txt") as f:
        corpus_contents = f.read()

    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


def test_tinystories_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = DATA_PATH / "tinystories_sample.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(
        corpus_contents, allowed_special={"<|endoftext|>"}
    )
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


def test_encode_special_token_trailing_newlines():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = DATA_PATH / "special_token_trailing_newlines.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(
        corpus_contents, allowed_special={"<|endoftext|>"}
    )
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


def test_encode_special_token_double_newline_non_whitespace():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = DATA_PATH / "special_token_double_newlines_non_whitespace.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(
        corpus_contents, allowed_special={"<|endoftext|>"}
    )
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


def test_encode_iterable_tinystories_sample_roundtrip():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(DATA_PATH / "tinystories_sample.txt") as f:
        all_ids = list(tokenizer.encode_iterable(f))
    with open(DATA_PATH / "tinystories_sample.txt") as f:
        corpus_contents = f.read()
    assert tokenizer.decode(all_ids) == corpus_contents


def test_encode_iterable_tinystories_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = DATA_PATH / "tinystories_sample.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(
        corpus_contents, allowed_special={"<|endoftext|>"}
    )
    with open(DATA_PATH / "tinystories_sample.txt") as f:
        all_ids = list(tokenizer.encode_iterable(f))
    assert all_ids == reference_ids

    assert tokenizer.decode(all_ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="rlimit support for non-linux systems is spotty.",
)
def test_encode_iterable_memory_usage():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(DATA_PATH / "tinystories_sample_5M.txt") as f:
        list(_encode_iterable(tokenizer, f))


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="rlimit support for non-linux systems is spotty.",
)
@pytest.mark.xfail(
    reason="Tokenizer.encode is expected to take more memory than allotted (1MB)."
)
def test_encode_memory_usage():
    """
    We expect this test to fail, since Tokenizer.encode is not expected to be memory efficient.
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(DATA_PATH / "tinystories_sample_5M.txt") as f:
        contents = f.read()
        _ = _encode(tokenizer, contents)


@memory_limit(int(1e6))
def _encode_iterable(tokenizer, iterable):
    """
    We place tokenizer.encode_iterable into a separate function so we can limit memory
    for just this function. We set the memory limit to 1MB.
    """
    yield from tokenizer.encode_iterable(iterable)


@memory_limit(int(1e6))
def _encode(tokenizer, text):
    """
    We place tokenizer.encode into a separate function so we can limit memory
    for just this function. We set the memory limit to 1MB.
    """
    return tokenizer.encode(text)
