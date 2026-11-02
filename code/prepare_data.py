"""
Phase 0 data preparation (DECLARATIONS.md sections 3 and 4).

Stages, run in this order (``--stage all`` runs all three):

  split      Download raw documents, choose validation/test documents, write the
             train split, remove exact duplicates across splits and across corpora,
             and write data/manifest.json.
  tokenizer  Train the byte-level BPE tokenizer on equal-size samples of the three
             training corpora (train splits only).
  tokenize   Encode every split into uint16 token files, with <|endoftext|> after
             each document, and check that tokens decode back to the exact text.

Design choices (each one addresses a specific risk):

* Dataset revisions are pinned. The Hugging Face commit hash of each dataset is
  recorded, and every file is downloaded at that revision, so a rerun reads the
  same bytes even if the dataset is updated.
* Validation/test documents are a uniform random sample of the first source file
  (several spread-out files for Wikipedia, whose files are ordered by article ID).
  The sample uses a keyed hash of each document, so it is reproducible and does not
  depend on how much training data is downloaded. Downloading more training data
  later only appends to the train split; validation and test never change.
* Exact duplicates are removed across splits AND across corpora. FineWeb-Edu is a
  filtered subset of FineWeb, so the same document can appear in both. A training
  document is dropped if its text exactly matches any validation/test document of
  any corpus. (Near-duplicate / 13-gram checking is a separate step before final
  evaluation, DECLARATIONS.md section 3.)
* The end-of-text token is NOT registered in the tokenizer. It is id
  vocab_size - 1, appended by this script. A document that literally contains the
  string "<|endoftext|>" is therefore encoded as ordinary text and can never
  produce a fake document boundary.
* For every token id, the number of UTF-8 bytes it represents is saved
  (token_bytes.npy). Bits-per-byte evaluation sums these over scored targets.
  This script checks, for every document, that the byte counts add up exactly to
  the document's UTF-8 length.

Usage (on Modal, see modal_app.py):
    modal run modal_app.py::prepare_data --stage all --train-mb 400

Local test with fake data:  LOCAL_SOURCE_DIR=/path python code/prepare_data.py ...
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import io
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Sources (DECLARATIONS.md section 3)
# ---------------------------------------------------------------------------
CORPORA = {
    "fineweb_edu": dict(repo="HuggingFaceFW/fineweb-edu", prefix="sample/10BT",
                        ext=".parquet", role="train", eval_files="first"),
    "fineweb": dict(repo="HuggingFaceFW/fineweb", prefix="sample/10BT",
                    ext=".parquet", role="train", eval_files="first"),
    "dclm": dict(repo="mlfoundations/dclm-baseline-1.0",
                 prefix="global-shard_01_of_10/local-shard_0_of_10",
                 ext=".jsonl.zst", role="train", eval_files="first"),
    # Out-of-domain evaluation only: no train split. Files are ordered by article
    # ID (older articles first), so eval documents come from 3 spread-out files.
    "wikipedia": dict(repo="wikimedia/wikipedia", prefix="20231101.en",
                      ext=".parquet", role="eval_only", eval_files="spread3"),
}
TRAIN_CORPORA = [c for c, s in CORPORA.items() if s["role"] == "train"]

EVAL_CANDIDATES = int(os.environ.get("EVAL_CANDIDATES", 40_000))      # documents kept by the bottom-k sample per corpus
TRAIN_SHARD_BYTES = 512 * 10**6
SCRIPT_VERSION = 1


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def doc_hash(text: str) -> bytes:
    """Exact-duplicate identity of a document (16-byte BLAKE2b of its UTF-8)."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def sample_key(h: bytes, seed: int) -> int:
    """Uniform pseudo-random key for reproducible bottom-k sampling."""
    return int.from_bytes(
        hashlib.blake2b(h, digest_size=8, key=seed.to_bytes(8, "little")).digest(), "little")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


# ---------------------------------------------------------------------------
# Source access (Hugging Face Hub, or a local directory for testing)
# ---------------------------------------------------------------------------
LOCAL_SOURCE_DIR = os.environ.get("LOCAL_SOURCE_DIR")


def resolve_revision(repo: str) -> str:
    if LOCAL_SOURCE_DIR:
        return "local-test"
    from huggingface_hub import HfApi
    return HfApi().dataset_info(repo).sha


def list_files(spec: dict, revision: str) -> list[dict]:
    """Sorted list of {path, size} under spec['prefix'] with the right extension."""
    if LOCAL_SOURCE_DIR:
        root = Path(LOCAL_SOURCE_DIR) / spec["repo"]
        files = [dict(path=str(p.relative_to(root)).replace(os.sep, "/"), size=p.stat().st_size)
                 for p in (root / spec["prefix"]).rglob("*") if p.name.endswith(spec["ext"])]
    else:
        from huggingface_hub import HfApi
        files = []
        for item in HfApi().list_repo_tree(spec["repo"], path_in_repo=spec["prefix"],
                                           repo_type="dataset", revision=revision,
                                           recursive=True):
            if getattr(item, "size", None) is not None and item.path.endswith(spec["ext"]):
                files.append(dict(path=item.path, size=item.size))
    files.sort(key=lambda f: f["path"])
    if not files:
        raise RuntimeError(f"No {spec['ext']} files under {spec['repo']}/{spec['prefix']}")
    return files


def fetch(spec: dict, revision: str, path: str, cache_dir: Path) -> Path:
    if LOCAL_SOURCE_DIR:
        return Path(LOCAL_SOURCE_DIR) / spec["repo"] / path
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(spec["repo"], path, repo_type="dataset",
                                revision=revision, cache_dir=str(cache_dir)))


def read_docs(local_path: Path, ext: str):
    """Yield the text of each document in one source file, in file order."""
    if ext == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(local_path)
        for batch in pf.iter_batches(batch_size=2048, columns=["text"]):
            for t in batch.column(0).to_pylist():
                if t:
                    yield t
    elif ext == ".jsonl.zst":
        import zstandard
        with open(local_path, "rb") as fh:
            reader = zstandard.ZstdDecompressor().stream_reader(fh)
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                t = json.loads(line).get("text")
                if t:
                    yield t
    else:
        raise ValueError(ext)


# ---------------------------------------------------------------------------
# Stage 1: splits
# ---------------------------------------------------------------------------
def eval_source_files(spec: dict, files: list[dict]) -> list[dict]:
    if spec["eval_files"] == "first":
        return files[:1]
    if spec["eval_files"] == "spread3":
        n = len(files)
        return [files[i] for i in sorted({0, n // 3, (2 * n) // 3})]
    raise ValueError(spec["eval_files"])


def select_eval(corpus, spec, revision, files, cache_dir, seed, eval_bytes):
    """Uniform random test/val documents via bottom-k sampling on a keyed hash."""
    src = eval_source_files(spec, files)
    heap: list[tuple[int, bytes, str]] = []   # max-heap on key via negation
    in_heap: set[bytes] = set()
    n_scanned = n_dup_in_source = 0
    seen: set[bytes] = set()
    for f in src:
        log(f"{corpus}: scanning {f['path']} for eval documents")
        for text in read_docs(fetch(spec, revision, f["path"], cache_dir), spec["ext"]):
            n_scanned += 1
            h = doc_hash(text)
            if h in seen:                 # exact duplicate inside the eval source
                n_dup_in_source += 1
                continue
            seen.add(h)
            k = sample_key(h, seed)
            if len(heap) < EVAL_CANDIDATES:
                heapq.heappush(heap, (-k, h, text)); in_heap.add(h)
            elif k < -heap[0][0]:
                _, old_h, _ = heapq.heapreplace(heap, (-k, h, text))
                in_heap.discard(old_h); in_heap.add(h)
    ranked = sorted(((-nk, h, t) for nk, h, t in heap))   # ascending key = random order
    test, val, nb_test, nb_val = [], [], 0, 0
    for _, h, t in ranked:
        b = len(t.encode("utf-8"))
        if nb_test < eval_bytes:
            test.append((h, t)); nb_test += b
        elif nb_val < eval_bytes:
            val.append((h, t)); nb_val += b
        else:
            break
    if nb_val < eval_bytes:
        raise RuntimeError(f"{corpus}: not enough documents for {eval_bytes} B test + val; "
                           f"raise EVAL_CANDIDATES or use more source files")
    stats = dict(eval_source_files=[f["path"] for f in src], docs_scanned=n_scanned,
                 exact_dups_in_eval_source=n_dup_in_source)
    return test, val, stats


def write_docs_jsonl(path: Path, docs) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = nbytes = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for _, t in docs:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            n += 1; nbytes += len(t.encode("utf-8"))
    return dict(file=str(path), docs=n, utf8_bytes=nbytes)


def write_train(corpus, spec, revision, files, cache_dir, out_dir, train_bytes,
                eval_hashes_all, own_eval_hashes):
    """Stream documents in file order into the train split until train_bytes."""
    shard_idx, shard_bytes, f_out = 0, 0, None
    shards, total_docs, total_bytes = [], 0, 0
    skipped_own_eval = skipped_dup_of_eval = within_train_dups = 0
    train_seen: set[bytes] = set()
    files_used = []
    done = False
    for f in files:
        if done:
            break
        log(f"{corpus}: train from {f['path']}")
        docs_from_file = 0
        for text in read_docs(fetch(spec, revision, f["path"], cache_dir), spec["ext"]):
            h = doc_hash(text)
            if h in eval_hashes_all:
                if h in own_eval_hashes:
                    skipped_own_eval += 1
                else:
                    skipped_dup_of_eval += 1
                continue
            if h in train_seen:
                within_train_dups += 1      # kept (declared: dedup across splits only)
            else:
                train_seen.add(h)
            if f_out is None or shard_bytes >= TRAIN_SHARD_BYTES:
                if f_out:
                    f_out.close()
                p = out_dir / f"train_{shard_idx:05d}.jsonl"
                shards.append(p); shard_idx += 1; shard_bytes = 0
                f_out = open(p, "w", encoding="utf-8", newline="\n")
            b = len(text.encode("utf-8"))
            f_out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            shard_bytes += b; total_bytes += b; total_docs += 1; docs_from_file += 1
            if total_bytes >= train_bytes:
                done = True
                break
        files_used.append(dict(path=f["path"], size=f["size"], docs_taken=docs_from_file,
                               whole_file=not done))
    if f_out:
        f_out.close()
    if total_bytes < train_bytes:
        raise RuntimeError(f"{corpus}: only {total_bytes} B available under the prefix")
    return dict(shards=[str(p) for p in shards], docs=total_docs, utf8_bytes=total_bytes,
                files_used=files_used, skipped_own_eval_docs=skipped_own_eval,
                skipped_exact_dup_of_any_eval_doc=skipped_dup_of_eval,
                within_train_exact_dups_kept=within_train_dups)


def stage_split(args) -> None:
    out = Path(args.out)
    cache_dir = out / "raw_cache"
    eval_bytes = int(args.eval_mb * 10**6)
    train_bytes = int(args.train_mb * 10**6)
    manifest = dict(script="code/prepare_data.py", script_version=SCRIPT_VERSION,
                    created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    args=vars(args), python=platform.python_version(), corpora={})

    # Pass 1: choose eval documents for every corpus (including Wikipedia).
    evals, meta = {}, {}
    pins = json.loads(args.pin_revisions) if args.pin_revisions else {}
    for corpus, spec in CORPORA.items():
        # Rebuild on another Modal account (D10): use the revision recorded in the original
        # manifest, NOT the dataset's current main branch, which may have moved since.
        rev = pins[corpus] if corpus in pins else resolve_revision(spec["repo"])
        if corpus in pins:
            log(f"{corpus}: revision pinned from original manifest")
        files = list_files(spec, rev)
        log(f"{corpus}: {spec['repo']}@{rev[:10]}, {len(files)} files under {spec['prefix']}")
        test, val, stats = select_eval(corpus, spec, rev, files, cache_dir, args.seed, eval_bytes)
        evals[corpus] = (test, val)
        meta[corpus] = dict(spec=spec, revision=rev, files=files, eval_stats=stats)

    # Within each corpus, test and val are disjoint by construction (distinct hashes).
    # Across corpora an eval document may repeat (harmless for evaluation); record it.
    eval_hashes_all: set[bytes] = set()
    cross_eval_overlap = 0
    for corpus, (test, val) in evals.items():
        for h, _ in test + val:
            if h in eval_hashes_all:
                cross_eval_overlap += 1
            eval_hashes_all.add(h)
    manifest["eval_docs_shared_between_corpora"] = cross_eval_overlap

    # Pass 2: write eval splits, then train splits.
    for corpus, spec in CORPORA.items():
        rev = meta[corpus]["revision"]
        test, val = evals[corpus]
        d = out / "text" / corpus
        entry = dict(repo=spec["repo"], prefix=spec["prefix"], revision=rev, role=spec["role"],
                     n_source_files_available=len(meta[corpus]["files"]),
                     **meta[corpus]["eval_stats"])
        entry["test"] = write_docs_jsonl(d / "test.jsonl", test)
        entry["val"] = write_docs_jsonl(d / "val.jsonl", val)
        if spec["role"] == "train":
            own = {h for h, _ in test + val}
            entry["train"] = write_train(corpus, spec, rev, meta[corpus]["files"], cache_dir, d,
                                         train_bytes, eval_hashes_all, own)
        manifest["corpora"][corpus] = entry
        log(f"{corpus}: test {entry['test']['docs']} docs, val {entry['val']['docs']} docs"
            + (f", train {entry['train']['docs']} docs / {entry['train']['utf8_bytes']/1e6:.0f} MB"
               if "train" in entry else ""))

    # File hashes. The test hashes are what DECLARATIONS.md section 3 locks.
    for corpus, entry in manifest["corpora"].items():
        for split in ("test", "val"):
            entry[split]["sha256"] = sha256_file(Path(entry[split]["file"]))
        if "train" in entry:
            entry["train"]["shard_sha256"] = [sha256_file(Path(p)) for p in entry["train"]["shards"]]
    write_json(out / "manifest.json", manifest)
    log("split stage done; manifest written")


# ---------------------------------------------------------------------------
# Stage 2: tokenizer (DECLARATIONS.md section 4)
# ---------------------------------------------------------------------------
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte <-> printable-unicode map used by byte-level BPE."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


def load_manifest(out: Path) -> dict:
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def iter_train_texts(entry: dict, max_bytes: int):
    """First max_bytes of a corpus's train split (stable when more data is added)."""
    nb = 0
    for shard in entry["train"]["shards"]:
        for t in iter_jsonl(Path(shard)):
            yield t
            nb += len(t.encode("utf-8"))
            if nb >= max_bytes:
                return


def stage_tokenizer(args) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    out = Path(args.out)
    manifest = load_manifest(out)
    per_corpus = int(args.tok_sample_mb * 10**6)
    for c in TRAIN_CORPORA:
        avail = manifest["corpora"][c]["train"]["utf8_bytes"]
        if avail < per_corpus:
            raise RuntimeError(f"{c}: train has {avail} B < tokenizer sample {per_corpus} B")

    def texts():
        for c in TRAIN_CORPORA:                 # equal-size sample of each train split
            log(f"tokenizer: reading {per_corpus/1e6:.0f} MB of {c} train")
            yield from iter_train_texts(manifest["corpora"][c], per_corpus)

    n_bpe = args.vocab_size - 1                 # last id is reserved for <|endoftext|>
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=n_bpe, special_tokens=[],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    t0 = time.time()
    tok.train_from_iterator(texts(), trainer)
    log(f"tokenizer trained in {time.time()-t0:.0f}s")
    if tok.get_vocab_size() != n_bpe:
        raise RuntimeError(f"BPE vocab {tok.get_vocab_size()} != {n_bpe}")

    # Bytes represented by each token id; <|endoftext|> represents 0 bytes.
    byte_decoder = {u: b for b, u in bytes_to_unicode().items()}
    if set(byte_decoder) != set(pre_tokenizers.ByteLevel.alphabet()):
        raise RuntimeError("byte-level alphabet mismatch")
    token_bytes = np.zeros(args.vocab_size, dtype=np.int16)
    for i in range(n_bpe):
        token_bytes[i] = len(bytes(byte_decoder[ch] for ch in tok.id_to_token(i)))
    if token_bytes[:n_bpe].min() < 1:
        raise RuntimeError("a BPE token represents zero bytes")

    d = out / "tokenizer"
    d.mkdir(parents=True, exist_ok=True)
    tok.save(str(d / "tokenizer.json"))
    np.save(d / "token_bytes.npy", token_bytes)
    meta = dict(type="byte-level BPE (HF tokenizers)", vocab_size=args.vocab_size,
                bpe_vocab_size=n_bpe, eot_id=args.vocab_size - 1,
                eot_note="<|endoftext|> is not a tokenizer token; id appended by prepare_data.py",
                trained_on={c: per_corpus for c in TRAIN_CORPORA},
                tokenizers_version=__import__("tokenizers").__version__,
                tokenizer_json_sha256=sha256_file(d / "tokenizer.json"),
                token_bytes_sha256=sha256_file(d / "token_bytes.npy"))
    write_json(d / "meta.json", meta)
    manifest["tokenizer"] = meta
    write_json(out / "manifest.json", manifest)
    log("tokenizer stage done")


# ---------------------------------------------------------------------------
# Stage 3: tokenize (uint16 token files)
# ---------------------------------------------------------------------------
def encode_split(tok, token_bytes, eot_id, texts, out_path: Path, n_roundtrip=200):
    """Encode documents, append EOT after each, verify bytes and round-trip."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_docs = n_tokens = n_bytes = 0
    batch: list[str] = []
    with open(out_path, "wb") as f:
        def flush():
            nonlocal n_docs, n_tokens, n_bytes
            encs = tok.encode_batch(batch, add_special_tokens=False)
            parts = []
            for text, enc in zip(batch, encs):
                ids = np.asarray(enc.ids, dtype=np.int64)
                ub = len(text.encode("utf-8"))
                if int(token_bytes[ids].sum()) != ub:          # exact BPB denominator
                    raise RuntimeError(f"byte count mismatch in {out_path}, doc {n_docs}")
                if n_docs < n_roundtrip and tok.decode(enc.ids, skip_special_tokens=False) != text:
                    raise RuntimeError(f"round-trip decode failed in {out_path}, doc {n_docs}")
                parts.append(ids); parts.append(np.array([eot_id]))
                n_docs += 1; n_tokens += len(ids) + 1; n_bytes += ub
            arr = np.concatenate(parts)
            if arr.max() >= 2**16:
                raise RuntimeError("token id does not fit uint16")
            arr.astype(np.uint16).tofile(f)
            batch.clear()
        for t in texts:
            batch.append(t)
            if len(batch) >= 4096:
                flush()
        if batch:
            flush()
    return dict(file=str(out_path), docs=n_docs, tokens_incl_eot=n_tokens,
                tokens_excl_eot=n_tokens - n_docs, utf8_bytes=n_bytes,
                bytes_per_token=round(n_bytes / max(1, n_tokens - n_docs), 4),
                sha256=sha256_file(out_path))


def stage_tokenize(args) -> None:
    from tokenizers import Tokenizer
    out = Path(args.out)
    manifest = load_manifest(out)
    tdir = out / "tokenizer"
    tok = Tokenizer.from_file(str(tdir / "tokenizer.json"))
    token_bytes = np.load(tdir / "token_bytes.npy")
    eot_id = json.loads((tdir / "meta.json").read_text())["eot_id"]
    results = {}
    for corpus, entry in manifest["corpora"].items():
        results[corpus] = {}
        for split in ("test", "val", "train"):
            if split not in entry:
                continue
            if split == "train":
                texts = (t for s in entry["train"]["shards"] for t in iter_jsonl(Path(s)))
            else:
                texts = iter_jsonl(Path(entry[split]["file"]))
            log(f"tokenize: {corpus}/{split}")
            r = encode_split(tok, token_bytes, eot_id, texts,
                             out / "tokens" / corpus / f"{split}.bin")
            results[corpus][split] = r
            log(f"  {r['tokens_incl_eot']:,} tokens, {r['bytes_per_token']} bytes/token")
    manifest["tokens"] = dict(dtype="uint16", eot_after_each_doc=True, splits=results)
    write_json(out / "manifest.json", manifest)
    log("tokenize stage done")


# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, help="output root (Modal Volume mount)")
    p.add_argument("--stage", default="all", choices=["split", "tokenizer", "tokenize", "all"])
    p.add_argument("--train_mb", type=float, default=400,
                   help="train-split size per training corpus, MB of UTF-8 text")
    p.add_argument("--eval_mb", type=float, default=10, help="size of each val/test split, MB")
    p.add_argument("--tok_sample_mb", type=float, default=200,
                   help="tokenizer training text per training corpus, MB")
    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--pin_revisions", default="",
                   help='JSON {corpus: HF commit}; rebuild from these exact dataset versions')
    args = p.parse_args(argv)
    log(f"prepare_data v{SCRIPT_VERSION}: {vars(args)}")
    if args.stage in ("split", "all"):
        stage_split(args)
    if args.stage in ("tokenizer", "all"):
        stage_tokenizer(args)
    if args.stage in ("tokenize", "all"):
        stage_tokenize(args)


if __name__ == "__main__":
    main(sys.argv[1:])
