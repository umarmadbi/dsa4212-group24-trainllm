"""
Modal app for the Phase 5 vocabulary data (D31). CPU only. Run from the repo root.

Kept OUTSIDE code/ and modal_app.py on purpose: the training launcher fingerprints code/, configs/,
modal_app.py and requirements.txt (code_sha256). Phase 4 was still running when this was added, so its
later rounds keep exactly the same code fingerprint as their pilots. This app never writes outside
/data/vocab/ on the data Volume (see tools/prepare_vocab.py).

    modal run --detach tools/vocab_app.py::build          # 3 variants + the 8,192 reproduction check, in parallel
    modal run tools/vocab_app.py::fetch                   # ~30-60 min later: copy manifests/tokenizers into data/vocab/,
                                                          #   check the 8,192 reproduction, report nesting and bytes/token
    modal run tools/vocab_app.py::verify                  # hashes of every variant file AND every baseline 8k file

Same image (pinned requirements.txt) and same data Volume as modal_app.py.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("code", remote_path="/root/code")
    .add_local_dir("tools", remote_path="/root/tools")
)
data_vol = modal.Volume.from_name("small-llm-data", create_if_missing=False)
app = modal.App("dsa4212-vocab-build", image=image)

SIZES = "4096,16384,32768"
CHECK = "_check_v8192"


@app.function(cpu=8, memory=32768, timeout=4 * 60 * 60, volumes={"/data": data_vol})
def build_variant(vocab_size: int, tokenizer_only: bool = False, name: str | None = None) -> dict:
    import sys
    sys.path.insert(0, "/root/tools")
    import prepare_vocab
    data_vol.reload()
    vm = prepare_vocab.build("/data", vocab_size, tokenize=not tokenizer_only, name=name)
    data_vol.commit()
    return dict(name=name or f"v{vocab_size}", tokenizer_sha256=vm["tokenizer"]["tokenizer_json_sha256"],
                build_seconds=vm["build_seconds"])


@app.function(cpu=2, memory=4096, timeout=10 * 60, volumes={"/data": data_vol})
def read_variant(name: str) -> dict | None:
    d = Path("/data/vocab") / name
    if not (d / "tokenizer" / "meta.json").exists():
        return None
    out = {"tokenizer.json": (d / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8"),
           "meta.json": (d / "tokenizer" / "meta.json").read_text(encoding="utf-8")}
    if (d / "manifest.json").exists():
        out["manifest.json"] = (d / "manifest.json").read_text(encoding="utf-8")
    tmp = sorted(p.name for p in Path("/data/vocab").glob(f".tmp_{name}_*"))
    out["unfinished_builds"] = tmp
    return out


@app.function(cpu=4, memory=8192, timeout=60 * 60, volumes={"/data": data_vol})
def hash_files(paths: list[str]) -> dict:
    import hashlib
    res = {}
    for p in paths:
        f = Path(p)
        if not f.is_file():
            res[p] = None
            continue
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        res[p] = h.hexdigest()
    return res


def _workspace() -> str:
    return subprocess.run(["modal", "profile", "current"], capture_output=True, text=True).stdout.strip()


@app.local_entrypoint()
def build(vocab_sizes: str = SIZES, check: bool = True):
    """Spawn one CPU container per vocabulary (+ the 8,192 reproduction check). Needs --detach."""
    print(f"workspace profile: {_workspace()}")
    calls = []
    if check:
        calls.append((CHECK, build_variant.spawn(8192, tokenizer_only=True, name=CHECK)))
    for v in (int(x) for x in vocab_sizes.split(",")):
        calls.append((f"v{v}", build_variant.spawn(v)))
    for n, c in calls:
        print(f"  started {n}: call {c.object_id}")
    print("Builds run on Modal whether or not this computer stays online (launched with --detach).\n"
          "Then:  modal run tools/vocab_app.py::fetch   and   modal run tools/vocab_app.py::verify")


def _merges(tok_json: str) -> list:
    m = json.loads(tok_json)["model"]["merges"]
    return [tuple(x) if isinstance(x, list) else tuple(x.split(" ")) for x in m]


@app.local_entrypoint()
def fetch(vocab_sizes: str = SIZES):
    """Copy each finished variant's manifest + tokenizer into data/vocab/<name>/ (committed with the
    results, like data/manifest.json), check the 8,192 reproduction and report nesting / bytes per token."""
    print(f"workspace profile: {_workspace()}")
    base = json.loads(Path("data/manifest.json").read_text(encoding="utf-8"))
    base_tok = Path("data/tokenizer.json").read_text(encoding="utf-8")
    chk = read_variant.remote(CHECK)
    if chk is None:
        print(f"{CHECK}: not finished yet")
    else:
        got = json.loads(chk["meta.json"])["tokenizer_json_sha256"]
        ok = got == base["tokenizer"]["tokenizer_json_sha256"]
        print(f"8,192 REPRODUCTION CHECK: {'PASSED' if ok else 'FAILED'} (builder {got[:12]}, Phase 0 "
              f"{base['tokenizer']['tokenizer_json_sha256'][:12]})"
              + ("" if ok else "  -> do NOT use the variants until this is understood"))
    merges = {8192: _merges(base_tok)}
    for v in (int(x) for x in vocab_sizes.split(",")):
        name = f"v{v}"
        r = read_variant.remote(name)
        if r is None or "manifest.json" not in r:
            print(f"{name}: not finished yet" + (f" (in progress: {r['unfinished_builds']})" if r else ""))
            continue
        d = Path("data/vocab") / name
        d.mkdir(parents=True, exist_ok=True)
        for k in ("manifest.json", "tokenizer.json", "meta.json"):
            (d / k).write_text(r[k], encoding="utf-8", newline="\n")
        vm = json.loads(r["manifest.json"])
        merges[v] = _merges(r["tokenizer.json"])
        bpt = {f"{c}/{s}": e["bytes_per_token"] for c, sp in vm["tokens"]["splits"].items() for s, e in sp.items()}
        print(f"{name}: fetched to {d}; bytes/token " + ", ".join(f"{k} {x}" for k, x in bpt.items()))
        print(f"   FineWeb-Edu train tokens {vm['tokens']['splits']['fineweb_edu']['train']['tokens_incl_eot']:,} "
              f"(8k: {base['tokens']['splits']['fineweb_edu']['train']['tokens_incl_eot']:,}); build {vm['build_seconds']} s")
    sizes = sorted(merges)
    for a, b in zip(sizes, sizes[1:]):
        same = merges[b][:len(merges[a])] == merges[a]
        print(f"nesting: the {a} merges {'ARE' if same else 'are NOT'} the first {len(merges[a])} merges of {b} "
              "(descriptive: same recipe, larger vocab = more merges)")


@app.local_entrypoint()
def verify(vocab_sizes: str = SIZES):
    """Every variant file against its fetched manifest, and every BASELINE 8k file against data/manifest.json
    (proves the build did not touch the data Phases 0-4 use)."""
    print(f"workspace profile: {_workspace()}")
    m = json.loads(Path("data/manifest.json").read_text(encoding="utf-8"))
    exp = {"/data/tokenizer/tokenizer.json": m["tokenizer"]["tokenizer_json_sha256"],
           "/data/tokenizer/token_bytes.npy": m["tokenizer"]["token_bytes_sha256"]}
    for sp in m["tokens"]["splits"].values():
        for e in sp.values():
            exp[e["file"]] = e["sha256"]
    n_base = len(exp)
    for v in (int(x) for x in vocab_sizes.split(",")):
        f = Path("data/vocab") / f"v{v}" / "manifest.json"
        if not f.exists():
            print(f"v{v}: no local manifest (run fetch first)")
            continue
        vm = json.loads(f.read_text(encoding="utf-8"))
        exp[f"/data/vocab/v{v}/tokenizer/tokenizer.json"] = vm["tokenizer"]["tokenizer_json_sha256"]
        exp[f"/data/vocab/v{v}/tokenizer/token_bytes.npy"] = vm["tokenizer"]["token_bytes_sha256"]
        for sp in vm["tokens"]["splits"].values():
            for e in sp.values():
                exp[e["file"]] = e["sha256"]
    got = hash_files.remote(sorted(exp))
    bad = [p for p in exp if got.get(p) != exp[p]]
    for p in sorted(exp):
        print(f"{'OK' if p not in bad else ('MISSING' if got.get(p) is None else 'MISMATCH'):8s} {p}")
    base_bad = [p for p in bad if not p.startswith("/data/vocab/")]
    print(f"baseline 8k files: {n_base - len(base_bad)}/{n_base} unchanged"
          + ("" if not base_bad else "  !!! BASELINE DATA CHANGED - stop all runs on this workspace"))
    print("VOCAB DATA CHECK " + ("PASSED" if not bad else f"FAILED ({len(bad)} files)"))
