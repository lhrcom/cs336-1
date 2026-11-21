"""Save / load a BPE vocab and merges in the GPT-2 format (vocab.json + merges.txt).

Each byte is mapped to a printable unicode character (space -> 'Ġ', newline -> 'Ċ'),
so tokens never contain a literal space or newline and the files are human-readable.
"""

import json
import os
from functools import lru_cache


@lru_cache
def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


@lru_cache
def unicode_to_bytes() -> dict[str, int]:
    return {c: b for b, c in bytes_to_unicode().items()}


def encode_token(tok: bytes) -> str:
    table = bytes_to_unicode()
    return "".join(table[b] for b in tok)


def decode_token(s: str) -> bytes:
    table = unicode_to_bytes()
    return bytes(table[c] for c in s)


def save_bpe(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    vocab_path = os.path.join(out_dir, "vocab.json")
    merges_path = os.path.join(out_dir, "merges.txt")

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({encode_token(tok): i for i, tok in vocab.items()}, f, ensure_ascii=False, indent=2)

    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{encode_token(a)} {encode_token(b)}\n")

    return vocab_path, merges_path


def load_bpe(vocab_path: str, merges_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(vocab_path, encoding="utf-8") as f:
        vocab = {i: decode_token(s) for s, i in json.load(f).items()}

    merges = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#version"):
                continue
            a, b = line.split(" ")
            merges.append((decode_token(a), decode_token(b)))

    return vocab, merges
