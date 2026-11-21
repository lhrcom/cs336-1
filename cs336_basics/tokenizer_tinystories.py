import time
from cs336_basics.bpe_io import save_bpe
from cs336_basics.tokenization import longest_token, train_bpe


def train_bpe_tinystories():
    text_path = "data/TinyStoriesV2-GPT4-train.txt"
    time_start = time.perf_counter()
    vocab, merges = train_bpe(text_path, vocab_size=10000, special_tokens=["<|endoftext|>"])
    time_end = time.perf_counter()
    total_time = time_end - time_start
    vocab_path, merges_path = save_bpe(vocab, merges, "data/tinystories_bpe")
    token_id, token = longest_token(vocab)
    print(f"Total time: {total_time} seconds")
    print(f"Saved vocab to {vocab_path}, merges to {merges_path}")
    print(f"Longest token: id={token_id}, {len(token)} bytes, {token.decode('utf-8', errors='replace')!r}")


if __name__ == "__main__":
    train_bpe_tinystories()
