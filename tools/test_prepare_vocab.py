"""
CPU test for tools/prepare_vocab.py (D31), on fake data:
  1. at the SAME vocabulary size the builder reproduces the Phase 0 pipeline (code/prepare_data.py stages
     tokenizer + tokenize) byte for byte: tokenizer.json, token_bytes.npy and every token file;
  2. it never touches the baseline files (hashes of data/tokenizer, data/tokens, data/manifest.json unchanged);
  3. same documents and bytes per split as the baseline; exact byte counts (checked inside encode_split);
  4. it refuses to overwrite an existing variant and refuses a different tokenizer sample size;
  5. FineWeb-Edu gets train/val/test, the other corpora only val/test.
    python tools/test_prepare_vocab.py
"""
import hashlib
import json
import random
import shutil
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "tools"))
import prepare_data as P  # noqa: E402
import prepare_vocab as V  # noqa: E402

_r = random.Random(1)
WORDS = ["".join(_r.choice("abcdefghijklmnopqrstuvwxyzéü") for _ in range(_r.randint(2, 10))) for _ in range(3000)] + ["数据", "модель"]


def fake_docs(rng, n):
    return [" ".join(rng.choice(WORDS) for _ in range(rng.randint(5, 120))) + rng.choice([".", "!", "\n"])
            for _ in range(n)]


def make_base(root: Path, tok_mb: float):
    rng = random.Random(0)
    man = {"args": {"tok_sample_mb": tok_mb}, "corpora": {}}
    for c in ("fineweb_edu", "fineweb", "dclm", "wikipedia"):
        e = {}
        tdir = root / "text" / c
        tdir.mkdir(parents=True)
        for split in ("val", "test"):
            f = tdir / f"{split}.jsonl"
            P.write_docs_jsonl(f, ((None, d) for d in fake_docs(rng, 60)))
            e[split] = {"file": str(f)}
        if c != "wikipedia":
            shards, nb = [], 0
            for i in range(2):
                f = tdir / f"train_{i:05d}.jsonl"
                docs = fake_docs(rng, 300)
                P.write_docs_jsonl(f, ((None, d) for d in docs))
                shards.append(str(f)); nb += sum(len(d.encode()) for d in docs)
            e["train"] = {"shards": shards, "utf8_bytes": nb}
        man["corpora"][c] = e
    P.write_json(root / "manifest.json", man)
    a = Namespace(out=str(root), vocab_size=400, tok_sample_mb=tok_mb)
    P.stage_tokenizer(a)
    P.stage_tokenize(a)


def tree_hashes(root: Path, sub=("tokenizer", "tokens", "manifest.json")):
    out = {}
    for s in sub:
        p = root / s
        for f in ([p] if p.is_file() else sorted(p.rglob("*"))):
            if f.is_file():
                out[str(f.relative_to(root))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def main():
    tok_mb = 0.03
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "data"
        make_base(root, tok_mb)
        before = tree_hashes(root)
        base = json.loads((root / "manifest.json").read_text())

        # 1. same vocab -> identical to Phase 0
        V.build(str(root), 400, tok_mb, name="same400")
        v = root / "vocab" / "same400"
        for f in ("tokenizer/tokenizer.json", "tokenizer/token_bytes.npy"):
            assert (v / f).read_bytes() == (root / f).read_bytes(), f"{f} differs from Phase 0"
        vm = json.loads((v / "manifest.json").read_text())
        for c, sp in vm["tokens"]["splits"].items():
            for s, e in sp.items():
                assert (v / "tokens" / c / f"{s}.bin").read_bytes() == (root / "tokens" / c / f"{s}.bin").read_bytes(), (c, s)
                assert e["sha256"] == base["tokens"]["splits"][c][s]["sha256"], (c, s)
        print("PASS  same vocab: tokenizer, token_bytes and every token file identical to the Phase 0 pipeline")

        # 5. splits per corpus
        got = {c: sorted(sp) for c, sp in vm["tokens"]["splits"].items()}
        assert got == {"fineweb_edu": ["test", "train", "val"], "fineweb": ["test", "val"],
                       "dclm": ["test", "val"], "wikipedia": ["test", "val"]}, got
        print("PASS  FineWeb-Edu train/val/test, other corpora val/test only")

        # another size: builds, same docs/bytes, more merges, fewer tokens
        V.build(str(root), 900, tok_mb)
        w = json.loads((root / "vocab" / "v900" / "manifest.json").read_text())
        for c, sp in w["tokens"]["splits"].items():
            for s, e in sp.items():
                b = base["tokens"]["splits"][c][s]
                assert (e["docs"], e["utf8_bytes"]) == (b["docs"], b["utf8_bytes"])
                assert e["tokens_excl_eot"] <= b["tokens_excl_eot"], (c, s)
                assert e["file"] == str(root / "vocab" / "v900" / "tokens" / c / f"{s}.bin")
                assert Path(e["file"]).exists()
        m400 = json.loads((root / "tokenizer" / "tokenizer.json").read_text())["model"]["merges"]
        m900 = json.loads((root / "vocab" / "v900" / "tokenizer" / "tokenizer.json").read_text())["model"]["merges"]
        print(f"PASS  vocab 900: same documents and bytes, fewer tokens; manifest paths point at the final directory "
              f"(nesting of merges: {m900[:len(m400)] == m400}, descriptive)")

        # data root reached through a link (as /data on Modal): manifest paths must use the link, not the target
        link = Path(t) / "datalink"; link.symlink_to(root)
        V.build(str(link), 900, tok_mb, name="vialink")
        vl = json.loads((root / "vocab" / "vialink" / "manifest.json").read_text())
        files = {e["file"] for sp in vl["tokens"]["splits"].values() for e in sp.values()}
        assert all(f.startswith(str(link / "vocab" / "vialink")) for f in files), sorted(files)[:2]
        print("PASS  data root via a link: manifest records the link path (as /data/... on Modal)")

        # 2. baseline untouched
        assert tree_hashes(root) == before, "baseline files changed"
        assert not list((root / "vocab").glob(".tmp_*")), "temporary directory left behind"
        print("PASS  baseline tokenizer, token files and manifest unchanged; no temporary directories left")

        # 4. refusals
        for kw, why in ((dict(vocab_size=900), "overwrite"), (dict(vocab_size=700, tok_sample_mb=0.02), "sample size")):
            try:
                V.build(str(root), kw["vocab_size"], kw.get("tok_sample_mb", tok_mb))
            except RuntimeError as e:
                print(f"PASS  refuses ({why}): {str(e)[:70]}")
            else:
                raise AssertionError(f"did not refuse: {why}")
        try:
            V._inside(root / "tokens" / "x.bin", root / "vocab")
        except RuntimeError:
            print("PASS  refuses a path outside data/vocab/")
        else:
            raise AssertionError("wrote outside vocab/")
    print("\nall prepare_vocab checks passed")


if __name__ == "__main__":
    main()
