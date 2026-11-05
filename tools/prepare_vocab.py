"""
Phase 5 (vocabulary) data: tokenizer + token files for another vocabulary size, built NEXT TO the baseline
8,192-token data and never on top of it (D31).

Why a separate tool: code/prepare_data.py's tokenizer/tokenize stages write to <data>/tokenizer/ and
<data>/tokens/ and rewrite <data>/manifest.json, i.e. exactly the files every other experiment reads.
Running them with --vocab_size 16384 while Phase 4 runs would silently swap the data under any run that
had not started yet. This tool:

* writes ONLY under <data>/vocab/v<V>/ (every output path is checked; anything else raises);
* builds into a temporary directory and renames it to v<V>/ only when everything is done and checked,
  and refuses to touch an existing v<V>/ (a half-built variant can never be read as complete);
* reads the raw text already on the Volume (paths from <data>/manifest.json): no download, same documents,
  same splits as the baseline;
* trains the tokenizer with the IDENTICAL recipe as Phase 0 (DECLARATIONS sec. 4: byte-level BPE, the same
  equal-size 200 MB sample of each training corpus's train split, <|endoftext|> = last id appended by us,
  token_bytes for exact BPB). `check_8192` proves it: the builder's 8,192 tokenizer must be byte-identical
  to the Phase 0 tokenizer recorded in data/manifest.json;
* tokenizes FineWeb-Edu train/val/test (Phases 4-6 train on FineWeb-Edu only, R9) and the val/test
  splits of FineWeb, DCLM and Wikipedia (for the four-set average and the Phase 6 test scoring);
* writes v<V>/manifest.json in the same layout as the baseline manifest (tokenizer + tokens sections),
  plus the SHA-256 of the baseline manifest it was derived from.

Imports the Phase 0 helpers from code/prepare_data.py unchanged (that file is not modified).

Usage (Modal): see tools/vocab_app.py.  Local:  python tools/prepare_vocab.py --data D --vocab_size 4096
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import prepare_data as P  # noqa: E402  (Phase 0 helpers, unchanged)

BUILDER_VERSION = 1
TRAIN_SPLITS = {"fineweb_edu": ("test", "val", "train")}   # other corpora: eval splits only
EVAL_SPLITS = ("test", "val")


def _inside(path: Path, root: Path) -> Path:
    path, root = path.resolve(), root.resolve()
    if root not in path.parents and path != root:
        raise RuntimeError(f"refusing to write outside {root}: {path}")
    return path


def train_tokenizer(manifest: dict, vocab_size: int, tok_sample_mb: float, out_dir: Path) -> dict:
    """Phase 0 tokenizer stage (prepare_data.stage_tokenizer) with the output directory as a parameter.
    The trainer settings are copied verbatim; check_8192 / the local test prove the result is identical."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    per_corpus = int(tok_sample_mb * 10**6)
    for c in P.TRAIN_CORPORA:
        avail = manifest["corpora"][c]["train"]["utf8_bytes"]
        if avail < per_corpus:
            raise RuntimeError(f"{c}: train has {avail} B < tokenizer sample {per_corpus} B")

    def texts():
        for c in P.TRAIN_CORPORA:                 # equal-size sample of each train split
            P.log(f"tokenizer v{vocab_size}: reading {per_corpus/1e6:.0f} MB of {c} train")
            yield from P.iter_train_texts(manifest["corpora"][c], per_corpus)

    n_bpe = vocab_size - 1                        # last id is reserved for <|endoftext|>
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=n_bpe, special_tokens=[],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    t0 = time.time()
    tok.train_from_iterator(texts(), trainer)
    P.log(f"tokenizer v{vocab_size} trained in {time.time()-t0:.0f}s")
    if tok.get_vocab_size() != n_bpe:
        raise RuntimeError(f"BPE vocab {tok.get_vocab_size()} != {n_bpe}")
    byte_decoder = {u: b for b, u in P.bytes_to_unicode().items()}
    if set(byte_decoder) != set(pre_tokenizers.ByteLevel.alphabet()):
        raise RuntimeError("byte-level alphabet mismatch")
    token_bytes = np.zeros(vocab_size, dtype=np.int16)
    for i in range(n_bpe):
        token_bytes[i] = len(bytes(byte_decoder[ch] for ch in tok.id_to_token(i)))
    if token_bytes[:n_bpe].min() < 1:
        raise RuntimeError("a BPE token represents zero bytes")
    d = out_dir / "tokenizer"
    d.mkdir(parents=True, exist_ok=True)
    tok.save(str(d / "tokenizer.json"))
    np.save(d / "token_bytes.npy", token_bytes)
    meta = dict(type="byte-level BPE (HF tokenizers)", vocab_size=vocab_size,
                bpe_vocab_size=n_bpe, eot_id=vocab_size - 1,
                eot_note="<|endoftext|> is not a tokenizer token; id appended by prepare_data.py",
                trained_on={c: per_corpus for c in P.TRAIN_CORPORA},
                tokenizers_version=__import__("tokenizers").__version__,
                tokenizer_json_sha256=P.sha256_file(d / "tokenizer.json"),
                token_bytes_sha256=P.sha256_file(d / "token_bytes.npy"))
    P.write_json(d / "meta.json", meta)
    return meta


def build(data: str, vocab_size: int, tok_sample_mb: float = 200, tokenize: bool = True,
          name: str | None = None) -> dict:
    """Build <data>/vocab/<name or v<V>>/. Returns the variant manifest."""
    from tokenizers import Tokenizer
    data = Path(data)
    base_manifest_path = data / "manifest.json"
    base = P.load_manifest(data)
    if base["tokenizer"]["trained_on"] != {c: int(tok_sample_mb * 10**6) for c in P.TRAIN_CORPORA}:
        raise RuntimeError(f"tokenizer sample differs from Phase 0: {base['tokenizer']['trained_on']}")
    if vocab_size > 2**16:
        raise RuntimeError("vocab does not fit uint16")
    root = data / "vocab"
    final = _inside(root / (name or f"v{vocab_size}"), root)
    if final.exists():
        raise RuntimeError(f"{final} already exists; refusing to overwrite (move it aside to rebuild)")
    tmp = _inside(root / f".tmp_{final.name}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}", root)
    tmp.mkdir(parents=True)
    t0 = time.time()
    meta = train_tokenizer(base, vocab_size, tok_sample_mb, tmp)
    results = {}
    if tokenize:
        tok = Tokenizer.from_file(str(tmp / "tokenizer" / "tokenizer.json"))
        token_bytes = np.load(tmp / "tokenizer" / "token_bytes.npy")
        for corpus, entry in base["corpora"].items():
            results[corpus] = {}
            for split in TRAIN_SPLITS.get(corpus, EVAL_SPLITS):
                if split not in entry:
                    continue
                texts = ((t for s in entry["train"]["shards"] for t in P.iter_jsonl(Path(s))) if split == "train"
                         else P.iter_jsonl(Path(entry[split]["file"])))
                P.log(f"tokenize v{vocab_size}: {corpus}/{split}")
                out = _inside(tmp / "tokens" / corpus / f"{split}.bin", root)
                r = P.encode_split(tok, token_bytes, meta["eot_id"], texts, out)
                base_r = base["tokens"]["splits"][corpus][split]
                if r["docs"] != base_r["docs"] or r["utf8_bytes"] != base_r["utf8_bytes"]:
                    raise RuntimeError(f"{corpus}/{split}: documents/bytes differ from the baseline "
                                       f"({r['docs']}/{r['utf8_bytes']} vs {base_r['docs']}/{base_r['utf8_bytes']})")
                r["file"] = str(final / "tokens" / corpus / f"{split}.bin")   # path after the rename
                results[corpus][split] = r
                P.log(f"  {r['tokens_incl_eot']:,} tokens, {r['bytes_per_token']} bytes/token "
                      f"(baseline 8k: {base_r['bytes_per_token']})")
    vm = dict(builder="tools/prepare_vocab.py", builder_version=BUILDER_VERSION,
              created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              build_seconds=round(time.time() - t0),
              derived_from=dict(manifest=str(base_manifest_path),
                                manifest_sha256=P.sha256_file(base_manifest_path),
                                baseline_tokenizer_sha256=base["tokenizer"]["tokenizer_json_sha256"]),
              corpora=base["corpora"], tokenizer=meta,
              tokens=dict(dtype="uint16", eot_after_each_doc=True, splits=results))
    P.write_json(tmp / "manifest.json", vm)
    tmp.rename(final)                              # only a complete, checked build gets the final name
    P.log(f"variant {final.name} done in {time.time()-t0:.0f}s")
    return vm


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build another vocabulary's data next to the baseline (D31)")
    p.add_argument("--data", required=True, help="data root (Modal Volume mount), e.g. /data")
    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument("--tok_sample_mb", type=float, default=200)
    p.add_argument("--tokenizer_only", action="store_true", help="train the tokenizer only (check_8192)")
    p.add_argument("--name", default=None, help="variant directory name (default v<V>)")
    a = p.parse_args(argv)
    build(a.data, a.vocab_size, a.tok_sample_mb, tokenize=not a.tokenizer_only, name=a.name)


if __name__ == "__main__":
    main()
