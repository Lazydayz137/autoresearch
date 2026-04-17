"""
Train a 32k BPE tokenizer on the MTG corpus shards.

Reads text from parquet shards in ~/.cache/autoresearch/mtg/data/ and writes
tokenizer artifacts to ~/.cache/autoresearch/mtg/tokenizer/:
    - tokenizer.pkl   (pickled tiktoken.Encoding - required format for autoresearch)
    - token_bytes.pt  (per-token utf-8 byte lengths for BPB eval)

Mirrors the conventions used by autoresearch/prepare.py (same SPLIT_PATTERN,
same special-token scheme, same parquet schema with a `text` column).

Usage:
    uv run prepare_mtg_tokenizer.py
"""
from __future__ import annotations

import os
import pickle  # mandatory: autoresearch train.py loads tokenizer.pkl via pickle.load
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

MTG_ROOT = Path(os.path.expanduser("~/.cache/autoresearch/mtg"))
DATA_DIR = MTG_ROOT / "data"
TOKENIZER_DIR = MTG_ROOT / "tokenizer"

VOCAB_SIZE = 32_000  # per Mission B1 brief (autoresearch default is 8192)

SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}|"""
    r""" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"


def list_shards() -> list[Path]:
    return sorted(p for p in DATA_DIR.iterdir()
                  if p.suffix == ".parquet" and not p.name.endswith(".tmp"))


def text_iter(val_shard: Path, doc_cap: int = 10_000,
              max_chars: int = 10_000_000_000):
    nchars = 0
    for path in list_shards():
        if path == val_shard:
            continue
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                if text is None:
                    continue
                doc = text if len(text) <= doc_cap else text[:doc_cap]
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def main() -> int:
    shards = list_shards()
    if len(shards) < 2:
        print(f"need >= 2 shards, got {len(shards)} at {DATA_DIR}", file=sys.stderr)
        return 2
    val_shard = shards[-1]
    print(f"Using {len(shards)} shards ({len(shards)-1} train + 1 val={val_shard.name})")

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    pkl_path = TOKENIZER_DIR / "tokenizer.pkl"
    token_bytes_path = TOKENIZER_DIR / "token_bytes.pt"

    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    print(f"Training rustbpe (target vocab={VOCAB_SIZE}, merges={vocab_size_no_special})...")
    t0 = time.time()
    tokenizer = rustbpe.Tokenizer()
    tokenizer.train_from_iterator(text_iter(val_shard), vocab_size_no_special,
                                  pattern=SPLIT_PATTERN)
    t1 = time.time()
    print(f"rustbpe train: {t1 - t0:.1f}s")

    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}

    enc = tiktoken.Encoding(
        name="mtg-rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(pkl_path, "wb") as f:
        pickle.dump(enc, f)
    print(f"Saved {pkl_path} (vocab={enc.n_vocab})")

    special_set = set(SPECIAL_TOKENS)
    lengths = []
    for token_id in range(enc.n_vocab):
        s = enc.decode([token_id])
        lengths.append(0 if s in special_set else len(s.encode("utf-8")))
    torch.save(torch.tensor(lengths, dtype=torch.int32), token_bytes_path)
    print(f"Saved {token_bytes_path}")

    # Per-source roundtrip checks
    wanted = [
        ("decklist", ("Event: ",)),
        ("article", ("Article: ",)),
        ("meta_snapshot", ("Metagame Snapshot: ",)),
        ("signal", ("Trading Signal ",)),
        ("social", ("reddit post", "twitter post", "mastodon post",
                    "bluesky post", "discord post")),
        ("tournament", ("Tournament: ",)),
        ("market_analysis", ("Market Analysis ",)),
    ]
    found: dict[str, str] = {}

    for path in shards:
        if len(found) == len(wanted):
            break
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                if not text:
                    continue
                for name, prefixes in wanted:
                    if name in found:
                        continue
                    if any(text.startswith(p) for p in prefixes):
                        found[name] = text
                if len(found) == len(wanted):
                    break
            if len(found) == len(wanted):
                break

    print("\n=== Per-source roundtrip + first 100 tokens ===\n")
    for name, _ in wanted:
        doc = found.get(name)
        if not doc:
            print(f"[{name}] MISSING in corpus")
            continue
        ids = enc.encode_ordinary(doc)
        dec = enc.decode(ids)
        assert dec == doc, f"roundtrip failed for {name}"
        first_line = doc.splitlines()[0] if doc else ""
        print(f"--- {name} (doc_bytes={len(doc.encode('utf-8'))}, tokens={len(ids)}) ---")
        print(f"first line: {first_line[:180]}")
        print(f"first 100 tokens: {ids[:100]}")
        print()

    # Coverage stats
    sample_chars = sample_toks = 0
    pf = pq.ParquetFile(val_shard)
    rg = pf.read_row_group(0)
    texts = rg.column("text").to_pylist()[:10_000]
    sample_chars = sum(len(t) for t in texts if t)
    sample_toks = sum(len(enc.encode_ordinary(t)) for t in texts if t)
    ratio = (sample_chars / sample_toks) if sample_toks else 0.0
    print(f"Val 10k-doc sample: {sample_chars:,} chars / {sample_toks:,} tokens = "
          f"{ratio:.2f} chars/token")
    return 0


if __name__ == "__main__":
    sys.exit(main())
