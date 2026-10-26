import cProfile
import pathlib

from nanollm.tokenizer.train_tokenizer import TokenizerTrainer

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "data"


def main():
    input_path = DATA_PATH / "big" / "TinyStoriesV2-GPT4-train.txt"
    profiler = cProfile.Profile()
    trainer = TokenizerTrainer(
        input_path=input_path,
        vocab_size=500,
        special_tokens=[b"<|endoftext|>"],
    )
    profiler.enable()
    trainer.train()
    profiler.disable()
    profiler.dump_stats("train_bpe.prof")


if __name__ == "__main__":
    main()
