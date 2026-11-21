from cs336_basics.tokenization import Tokenizer
import numpy as np
from pathlib import Path
base = Path(__file__).parent.parent
data_base_path = base / "data"
path_1 = data_base_path / "owt_train.txt"
path_2 = data_base_path / "owt_valid.txt"
path_3 = data_base_path / "TinyStoriesV2-GPT4-train.txt"
path_4 = data_base_path / "TinyStoriesV2-GPT4-valid.txt"


def encode_training_vaild():
    tok_1 = Tokenizer.from_files(
        data_base_path / "tinystories_bpe/vocab.json",
        data_base_path / "tinystories_bpe/merges.txt",
        special_tokens=["<|endoftext|>"]
    )

    tok_2 = Tokenizer.from_files(
                data_base_path / "owt_bpe/vocab.json",
                data_base_path / "owt_bpe/merges.txt",
                special_tokens=["<|endoftext|>"]
        )
    
    with open(path_1, "r", encoding="utf-8") as f:
            text_1 = f.read()
    with open(path_2, "r", encoding="utf-8") as f:
            text_2 = f.read()
    with open(path_3, "r", encoding="utf-8") as f:
            text_3 = f.read()
    with open(path_4, "r", encoding="utf-8") as f:
            text_4 = f.read()

    ids_1 = tok_1.encode(text_3)
    ids_2 = tok_1.encode(text_4)
    ids_3 = tok_2.encode(text_1)
    ids_4 = tok_2.encode(text_2)

    arr_1 = np.array(ids_1, dtype=np.uint16)
    arr_2 = np.array(ids_2, dtype=np.uint16)
    arr_3 = np.array(ids_3, dtype=np.uint16)
    arr_4 = np.array(ids_4, dtype=np.uint16)

    np.save(data_base_path / "tinystories_train.npy", arr_1)
    np.save(data_base_path / "tinystories_valid.npy", arr_2)
    np.save(data_base_path / "owt_train.npy", arr_3)
    np.save(data_base_path / "owt_valid.npy", arr_4)

if __name__ == "__main__":
    encode_training_vaild()