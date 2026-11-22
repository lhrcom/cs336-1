import random
import time 
from cs336_basics.paths import DATA_DIR
from cs336_basics.tokenization import Tokenizer

tok = Tokenizer.from_files(
        DATA_DIR / "tinystories_bpe/vocab.json",
        DATA_DIR / "tinystories_bpe/merges.txt",
        special_tokens=["<|endoftext|>"]
    )

def compression_ratio():
    random.seed(42)
    path = DATA_DIR / "owt_train.txt"
    
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    all_docs = [d for d in text.split("<|endoftext|>") if d.strip()]
    docs = random.sample(all_docs, 10)

    total_bytes  = sum(len(d.encode("utf-8")) for d in docs)
    total_tokens = sum(len(tok.encode(d)) for d in docs)
    print(total_bytes / total_tokens)

def tokenizer_throughput(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    n_bytes = len(text.encode("utf-8"))
    start  = time.perf_counter()
    ids = tok.encode(text)
    elapsed = time.perf_counter() - start
    throughput = n_bytes / elapsed
    print(f"{throughput:.0f} bytes/s  ({throughput/1e6:.2f} MB/s)")
    pile_bytes = 2e9
    seconds = pile_bytes / throughput
    print(f"{seconds/3600:.1f} 小时 = {seconds/86400:.1f} 天")
    

if __name__ == "__main__":
    path = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
    tokenizer_throughput(path)