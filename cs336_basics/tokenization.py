# BPE Tokenizer
#BPE Tokenizer Training

import heapq
import os
import time
import regex
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import BinaryIO
from collections.abc import Iterable, Iterator
import re  
import json
from cs336_basics.bpe_io import load_bpe

#Pre-tokenization

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Files smaller than this are pre-tokenized serially: starting worker
# processes costs more than it saves.
MIN_PARALLEL_BYTES = 1 << 20


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
    # Keep an overlap so a token straddling two mini chunks is still found
    overlap = len(split_special_token) - 1

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size + overlap)

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
            file.seek(initial_position)

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _pretokenize_chunk(args) -> Counter:
    """Worker: read bytes [start, end) of the file and count its pre-tokens."""
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    # Split on special tokens so no pre-token spans one of them
    if special_tokens:
        # Longest first, so overlapping special tokens match greedily
        ordered = sorted(special_tokens, key=len, reverse=True)
        pat = "|".join(re.escape(tok) for tok in ordered)
        parts = re.split(pat, text)
    else:
        parts = [text]

    return Counter(m.group() for part in parts for m in regex.finditer(PAT, part))


def pretokenize(input_path, special_tokens, num_processes=None) -> Counter:
    """Count pre-tokens in the file, using multiple processes for large files."""
    if num_processes is None:
        num_processes = os.cpu_count() or 1

    file_size = os.path.getsize(input_path)
    # Chunk boundaries must fall on a special token; without one, or for a
    # small file, use a single chunk.
    if num_processes <= 1 or not special_tokens or file_size < MIN_PARALLEL_BYTES:
        return _pretokenize_chunk((input_path, 0, file_size, special_tokens))

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))

    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    word_counts = Counter()
    with Pool(min(num_processes, len(tasks))) as pool:
        for counts in pool.imap_unordered(_pretokenize_chunk, tasks):
            word_counts.update(counts)
    return word_counts


class _ReversedPair:
    """Wraps a pair so that the lexicographically greater pair sorts first.

    heapq is a min-heap, but among equally frequent pairs BPE picks the
    greatest one, so the comparison is inverted.
    """

    __slots__ = ("pair",)

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair


def train_bpe(input_path, vocab_size=None, special_tokens=["<|endoftext|>"], num_processes=None):

    # Initialize vocabulary with single bytes and special tokens, vocab: vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    # Pre-tokenization: 
    '''
    Pre-tokenize the text into words using the regex pattern.
    Count the occurrences of each word.
    '''
    word_counts = pretokenize(input_path, special_tokens, num_processes)

    merges = []
    num_merges = vocab_size - len(vocab)

    # Each distinct pre-token becomes a list of tokens (initially single bytes).
    # words[i] is the current token sequence of word i, counts[i] its frequency.
    words = []
    counts = []
    for word, count in word_counts.items():
        words.append([bytes([x]) for x in word.encode("utf-8")])
        counts.append(count)

    # Count every adjacent pair once, and index which words contain each pair,
    # so a merge only has to touch the words it actually changes.
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)
    for idx, tokens in enumerate(words):
        count = counts[idx]
        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] += count
            pair_to_words[pair].add(idx)

    # Max-heap of (-count, pair) with lazy deletion: entries are never updated
    # in place. When a count changes a fresh entry is pushed, and an entry whose
    # count no longer matches pair_counts is discarded when it reaches the top.
    heap = [(-count, _ReversedPair(pair)) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    for _ in range(num_merges):
        # Most frequent pair; ties go to the lexicographically greater pair
        best_pair = None
        while heap:
            neg_count, entry = heap[0]
            if pair_counts.get(entry.pair) == -neg_count:
                best_pair = entry.pair
                break
            heapq.heappop(heap)  # stale entry
        if best_pair is None:
            break

        first, second = best_pair
        changed = set()  # pairs whose count changed during this merge
        new_token = first + second

        # Only words containing best_pair change
        for idx in pair_to_words.pop(best_pair):
            tokens = words[idx]
            count = counts[idx]

            # Remove this word's old pairs
            for pair in zip(tokens, tokens[1:]):
                pair_counts[pair] -= count
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                if pair != best_pair:
                    pair_to_words[pair].discard(idx)
                    changed.add(pair)

            # Merge every occurrence of best_pair in the word
            merged = []
            i = 0
            n = len(tokens)
            while i < n:
                if i + 1 < n and tokens[i] == first and tokens[i + 1] == second:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            words[idx] = merged

            # Add the word's new pairs
            for pair in zip(merged, merged[1:]):
                pair_counts[pair] += count
                pair_to_words[pair].add(idx)
                changed.add(pair)

        # Push one fresh heap entry per pair whose count changed
        for pair in changed:
            count = pair_counts.get(pair)
            if count:
                heapq.heappush(heap, (-count, _ReversedPair(pair)))

        merges.append(best_pair)
        vocab[len(vocab)] = new_token

    return vocab, merges



def longest_token(vocab: dict[int, bytes]) -> tuple[int, bytes]:
    """Return (token_id, token) for the longest token in the vocabulary, by byte length."""
    token_id = max(vocab, key=lambda i: len(vocab[i]))
    return token_id, vocab[token_id]

class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Special tokens missing from the vocab get new ids appended at the end
        existing = set(self.vocab.values())
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in existing:
                self.vocab[len(self.vocab)] = tok_bytes
                existing.add(tok_bytes)

        # bytes -> id, the inverse of vocab
        self.token_to_id = {tok: i for i, tok in self.vocab.items()}
        # pair -> rank; a lower rank means the merge was learned earlier
        self.merge_rank = {pair: rank for rank, pair in enumerate(merges)}

        # Regex that splits text on special tokens while keeping them (capture
        # group). Longest first, so overlapping special tokens match greedily.
        if self.special_tokens:
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pat = re.compile("(" + "|".join(re.escape(tok) for tok in ordered) + ")")
        else:
            self._special_pat = None
        self._special_set = set(self.special_tokens)

        self._pat = regex.compile(PAT)
        # pre-token string -> list of ids; pre-tokens repeat a lot in real text
        self._cache = {}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab, merges = load_bpe(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        """Apply the BPE merges to a single pre-token and return its ids."""
        cached = self._cache.get(pretoken)
        if cached is not None:
            return cached

        tokens = [bytes([b]) for b in pretoken.encode("utf-8")]
        merge_rank = self.merge_rank
        while len(tokens) > 1:
            # Among the pairs present, apply the one learned earliest
            best_pair = None
            best_rank = None
            for pair in zip(tokens, tokens[1:]):
                rank = merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair, best_rank = pair, rank
            if best_pair is None:
                break

            first, second = best_pair
            merged = []
            i = 0
            n = len(tokens)
            while i < n:
                if i + 1 < n and tokens[i] == first and tokens[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged

        ids = [self.token_to_id[tok] for tok in tokens]
        self._cache[pretoken] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        if self._special_pat is not None:
            parts = self._special_pat.split(text)
        else:
            parts = [text]

        ids = []
        for part in parts:
            if not part:
                continue
            if part in self._special_set:
                ids.append(self.token_to_id[part.encode("utf-8")])
                continue
            for m in self._pat.finditer(part):
                ids.extend(self._encode_pretoken(m.group()))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings (e.g. a file handle, line by
        line), so a large file never has to be held in memory at once."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        # Join bytes first: a single token may be only part of a UTF-8 character
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")


